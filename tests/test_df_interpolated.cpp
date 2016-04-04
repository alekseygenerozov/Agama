/** \file    test_df_interpolated.cpp
    \author  Eugene Vasiliev
    \date    April 2016

*/
#include "galaxymodel.h"
#include "df_halo.h"
#include "df_interpolated.h"
#include "potential_analytic.h"
#include "actions_spherical.h"
#include "math_core.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>


int main()
{
    df::DoublePowerLawParam paramDPL;
    paramDPL.alpha = 0;
    paramDPL.beta  = 7.5;
    paramDPL.j0    = 1;
    paramDPL.jcore = 0;
    paramDPL.ar    = 0.1;
    paramDPL.az    = (3-paramDPL.ar)/2;
    paramDPL.aphi  = paramDPL.az;
    paramDPL.br    = 1.2;
    paramDPL.bz    = 0.9;
    paramDPL.bphi  = 0.9;
    paramDPL.norm  = 78.19;
    potential::Plummer pot(1., 1.);
    const actions::ActionFinderSpherical af(pot);
    const df::DoublePowerLaw dfO(paramDPL);    // original distribution function
    unsigned int gridSize[3] = {40, 5, 5};
    df::InterpolatedDFParam paramInt = createInterpolatedDFParam(dfO, gridSize, 0.3, 5);
    const df::InterpolatedDF<1> dfI(paramInt); // interpolated DF
    std::cout << "M=" << pot.totalMass() << ", dfOrig M=" << dfO.totalMass() <<
        ", dfInter M=" << dfI.totalMass() << '\n';
    
    std::ofstream strm("test_df_interpolated.dfval");
    for(unsigned int i=0; i<gridSize[0]; i++) {
        for(unsigned int j=0; j<gridSize[1]; j++) {
            for(unsigned int k=0; k<gridSize[2]; k++) {
                strm << paramInt.gridJsum[i] << ' ' <<
                    paramInt.gridJrrel[j] << ' ' <<
                    paramInt.gridJphirel[k] << ' ' <<
                    paramInt.values[(i * gridSize[1] + j) * gridSize[2] + k] << '\n';
            }
            strm << '\n';
        }
        strm << '\n';
    }
    strm.close();

    strm.open("test_df_interpolated.densval");
    std::ofstream strmc("test_df_interpolated.denscomp");
    for(double r=0; r<20; r<1 ? r+=0.1 : r*=1.1) {
        coord::PosCyl point(r, 0, 0);
        std::vector<double> densArr =
            galaxymodel::computeMulticomponentDensity(pot, af, dfI, point, 1e-3, 10000);
        double densOrig, densInt, densSum=0;
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfO),
            point, 1e-3, 100000, &densOrig,NULL, NULL, NULL, NULL, NULL);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfI),
            point, 1e-3, 100000, &densInt, NULL, NULL, NULL, NULL, NULL);
        for(unsigned int i=0; i<densArr.size(); i++)
            densSum += densArr[i];

        strm << r << ' ' << pot.density(point) << ' ' <<
            densOrig << ' ' << densInt << ' ' << densSum << '\n';

        for(unsigned int i=0; i<gridSize[0]; i++)
            for(unsigned int j=0; j<gridSize[1]; j++)
                for(unsigned int k=0; k<gridSize[2]; k++) {
                    strmc << paramInt.gridJsum[i] << ' ' <<
                        paramInt.gridJrrel[j] << ' ' <<
                        paramInt.gridJphirel[k] << ' ' <<
                        densArr.at((i * gridSize[1] + j) * gridSize[2] + k) << '\n';
                }
        strmc << '\n';
    }

    return 0;
}
