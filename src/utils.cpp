#include "utils.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace utils {

/* -------- error reporting routines ------- */

namespace{  // internal

/// remove function signature from GCC __PRETTY_FUNCTION__
static std::string undecorateFunction(std::string origin)
{
#ifdef __GNUC__
    // parse the full function signature returned by __PRETTY_FUNCTION__
    std::string::size_type ind=origin.find('(');
    if(ind!=std::string::npos)
        origin.erase(ind);
    ind=origin.rfind(' ');
    if(ind!=std::string::npos)
        origin.erase(0, ind+1);
#endif
    return origin;
}

static char verbosityText(VerbosityLevel level)
{
    switch(level) {
        case VL_MESSAGE: return '.';
        case VL_WARNING: return '!';
        case VL_DEBUG:   return '-';
        case VL_VERBOSE: return '=';
        default: return '?';
    }
}

/// file to store the messages sent to msg() routine;
/// if not open, they are printed to stderr
static std::ofstream logfile;
    
/// read the environment variables controlling the verbosity level and log file redirection
static VerbosityLevel initVerbosityLevel()
{
    const char* env = std::getenv("LOGFILE");
    if(env) {
        logfile.open(env);
    }
    env = std::getenv("LOGLEVEL");
    if(env && env[0] >= '0' && env[0] <= '3')
        return static_cast<VerbosityLevel>(env[0]-'0');
    return VL_MESSAGE;  // default
}

/// default routine that dumps text messages to stderr
static void defaultmsg(VerbosityLevel level, const char* origin, const std::string &message)
{
    if(level > verbosityLevel)
        return;
    std::string msg = verbosityText(level) + 
        (origin==NULL || *origin=='\0' ? "" : "{"+undecorateFunction(origin)+"} ") + message + '\n';
    if(logfile.is_open())
        logfile << msg << std::flush;
    else
        std::cerr << msg << std::flush;
}
    
}  // namespace

/// global pointer to the routine that displays or logs information messages
MsgType* msg = &defaultmsg;

/// global variable controlling the verbosity of printout
VerbosityLevel verbosityLevel = initVerbosityLevel();

/* ----------- string/number conversion and parsing routines ----------------- */

int toInt(const char* val) {
    return strtol(val, NULL, 10);
}

float toFloat(const char* val) {
    return strtof(val, NULL);
}

double toDouble(const char* val) {
    return strtod(val, NULL);
}

std::string toString(double val, unsigned int width) {
    char buf[100];
    int len=snprintf(buf, 100, "%.*g", width, val);
    int offset=0;
    while(offset<len && buf[offset]==' ') offset++;
    return std::string(buf+offset);
}

std::string toString(float val, unsigned int width) {
    char buf[100];
    int len=snprintf(buf, 100, "%.*g", width, val);
    int offset=0;
    while(offset<len && buf[offset]==' ') offset++;
    return std::string(buf+offset);
}

std::string toString(int val) {
    char buf[100];
    snprintf(buf, 100, "%i", val);
    return std::string(buf);
}

std::string toString(unsigned int val) {
    char buf[100];
    snprintf(buf, 100, "%u", val);
    return std::string(buf);
}

std::string toString(const void* val) {
    char buf[100];
    snprintf(buf, 100, "%p", val);
    return std::string(buf);
}

bool toBool(const char* val) {
    return 
        strncmp(val, "yes", 3)==0 ||
        strncmp(val, "Yes", 3)==0 ||
        strncmp(val, "true", 4)==0 ||
        strncmp(val, "True", 4)==0 ||
        strncmp(val, "t", 1)==0 ||
        strncmp(val, "1", 1)==0;
}

//  Pretty-print - convert float (and integer) numbers to string of fixed width.
//  Employ sophisticated techniques to fit the number into a string of exactly the given length.
std::string pp(double num, unsigned int width)
{
    std::string result;
    if(num==0) { 
        for(int i=0; i<static_cast<int>(width)-1; i++) result+=' ';
        result+='0';
        return result;
    }
    unsigned int sign=num<0;
    double mag=log10(fabs(num));
    std::ostringstream stream;
    if(num!=num || num/2==num || num+0!=num)
    {
        if(width>=4) stream << std::setw(width) << num;
        else stream << "#ERR";
    }
    else if(width<=2+sign)  // display int if possible
    {
        if(mag<0) stream << (sign?"-":"+") << 0;
        else if(mag>=2-sign) stream << (sign?"-":"+") << "!";
        else stream << (int)floor(num+0.5);
    }
    else if(mag>=0 && mag+sign<width && mag<6)  // try fixed-point for |x|>=1
    {
        stream << std::setw(width) << std::setprecision(width-1-sign) << num;
        if(stream.str().find('e')!=std::string::npos) { 
            //std::string x=stream.str();
            //size_t e=x.find('e');
            stream.str(""); 
            stream << (int)floor(num+0.5); 
        }
    }
    else if(mag<0 && -mag+sign<width-2 && mag>=-4) // try fixed-point for |x|<1
    {
        stream << std::setw(width) << std::setprecision(width-1-sign+(int)floor(mag)) << num;
    }
    else
    {
        std::ostringstream strexp;
        int expon=static_cast<int>(floor(mag));
        strexp << std::setiosflags(std::ios_base::showpos) << expon;
        std::string expstr=strexp.str();
        size_t w=(width-expstr.size()-1);
        double mant=num*pow(10.0, -expon);
        if(w<sign)  // no luck with exp-format -- try fixed 
        {
            stream << (sign?"-":"+") << (mag<0 ? "0" : "!");
        }
        else 
        {
            if(w==sign) 
                stream << (sign?"-":""); // skip mantissa
            else if(w<=2+sign)
            { 
                int mantint=(int)floor(fabs(mant)+0.5);
                if(mantint>=10) mantint=9;  // hack
                stream << (sign?"-":"") << mantint;
            }
            else
                stream << std::setprecision(w-1-sign) << mant;
            stream << "e" << expstr;
        }
    }
    result=stream.str();
    // padding if necessary (add spaces in front of the string)
    if(result.length()<static_cast<size_t>(width))
        result.insert(0, width-result.length(), ' ');
    if(result.length()>static_cast<size_t>(width))  // cut tail if necessary (no warning given!)
        result=result.substr(0,width);
    return result;
}

std::vector<std::string> splitString(const std::string& src, const std::string& delim)
{
    std::vector<std::string> result;
    std::string str(src);
    std::string::size_type indx=str.find_first_not_of(delim);
    if(indx==std::string::npos) {
        result.push_back("");   // ensure that result contains at least one element
        return result;
    }
    if(indx>0)  // remove unnecessary delimiters at the beginning
        str=str.erase(0, indx);
    while(!str.empty()) {
        indx=str.find_first_of(delim);
        if(indx==std::string::npos)
            indx=str.size();
        result.push_back(str.substr(0, indx));
        str=str.erase(0, indx);
        indx=str.find_first_not_of(delim);
        if(indx==std::string::npos)
            break;
        str=str.erase(0, indx);
    }
    return result;
}

bool endsWithStr(const std::string& str, const std::string& end)
{
    return end.size()<=str.size() && str.find(end, str.size()-end.size())!=str.npos;
}

bool stringsEqual(const std::string& str1, const std::string& str2)
{
    std::string::size_type len=str1.size();
    if(len!=str2.size())
        return false;
    for(std::string::size_type i=0; i<len; i++)
        if(tolower(str1[i]) != tolower(str2[i]))
            return false;
    return true;
}

bool stringsEqual(const std::string& str1, const char* str2)
{
    if(str2==NULL)
        return false;
    for(std::string::size_type i=0; i<str1.size(); i++)
        if(str2[i]==0 || tolower(str1[i]) != tolower(str2[i]))
            return false;
    return str2[str1.size()]==0;  // ensure that the 2nd string length is the same as the 1st
}

}  // namespace
