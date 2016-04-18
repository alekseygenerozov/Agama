/** \file    test_df_interpolated.cpp
    \author  Eugene Vasiliev
    \date    April 2016

    Test the accuracy of linearly and cubically interpolated distribution functions,
    by comparing the total mass of the model and density values computed at several radii.
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

static actions::Actions unscaleJ(double s, double p, double q, double J0)
{
    const double Jsum = (exp(s) - 1) * J0;
    actions::Actions ac;
    ac.Jr   = Jsum * p;
    ac.Jphi = Jsum * (1-p) * (2*q-1);
    ac.Jz   = Jsum * (1-p) * (1-fabs(2*q-1));
    return ac;
}

int main()
{
    bool ok = true;
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
    df::InterpolatedDFParam paramL = df::createInterpolatedDFParam<1>(dfO, gridSize, 0.3, 5);
    const df::InterpolatedDF<1> dfL(paramL); // linearly-interpolated DF
    df::InterpolatedDFParam paramC = df::createInterpolatedDFParam<3>(dfO, gridSize, 0.3, 5);
    const df::InterpolatedDF<3> dfC(paramC); // cubic-interpolated DF
    std::cout << "Constructed interpolated DFs\n";
    std::ofstream strm("test_df_interpolated.dfval");
    for(unsigned int i=0; i<gridSize[0]; i++) {
        for(unsigned int j=0; j<gridSize[1]; j++) {
            for(unsigned int k=0; k<gridSize[2]; k++) {
                actions::Actions J = unscaleJ(paramL.gridJsum[i],
                    paramL.gridJrrel[j], paramL.gridJphirel[k], paramL.J0);
                double valL = dfL.value(J), valC = dfC.value(J);
                ok &= math::fcmp(valL, valC, 1e-12) == 0;  // should be the same
                strm <<
                    paramL.gridJsum[i] << ' ' <<
                    paramL.gridJrrel[j] << ' ' <<
                    paramL.gridJphirel[k] << ' ' <<
                    valL << ' ' << valC << '\n';
            }
        }
        strm << '\n';
    }
    strm.close();

    double sumLin=0;
    strm.open("test_df_interpolated.phasevolL");
    for(unsigned int i=0; i<paramL.amplitudes.size(); i++) {
        double phasevol = dfL.computePhaseVolume(i);
        strm << paramL.amplitudes[i] << ' ' << phasevol << '\n';
        sumLin += paramL.amplitudes[i] * phasevol;
    }
    strm.close();

    double sumCub=0;
    strm.open("test_df_interpolated.phasevolC");
    for(unsigned int i=0; i<paramC.amplitudes.size(); i++) {
        double phasevol = dfC.computePhaseVolume(i);
        strm << paramC.amplitudes[i] << ' ' << phasevol << '\n';
        sumCub += paramC.amplitudes[i] * phasevol;
    }
    strm.close();
    
    double massOrig = dfO.totalMass();
    double massLin  = dfL.totalMass();
    double massCub  = dfC.totalMass();
    std::cout << "M=" << pot.totalMass() << ", dfOrig M=" << massOrig <<
        ", dfInterLin M=" << massLin << ", sum over components=" << sumLin << 
        ", dfInterCub M=" << massCub << ", sum over components=" << sumCub << '\n';
    ok &= math::fcmp(massOrig, massLin, 1e-2)==0 && math::fcmp(sumLin, massLin, 1e-3)==0 &&
          math::fcmp(massOrig, massCub, 5e-3)==0 && math::fcmp(sumCub, massCub, 1e-3)==0;

    strm.open("test_df_interpolated.densval");
    std::ofstream strmL("test_df_interpolated.denscompL");
    std::ofstream strmC("test_df_interpolated.denscompC");
    for(double r=0; r<20; r<1 ? r+=0.1 : r*=1.1) {
        coord::PosCyl point(r, 0, 0);
        std::vector<double> densArrL(dfL.size()), densArrC(dfC.size());
        double densOrig, densIntL, densIntC;
        computeMoments(galaxymodel::GalaxyModelMulticomponent(pot, af, dfL),
            point, 1e-3, 10000, &densArrL[0], NULL, NULL, NULL, NULL, NULL);
        computeMoments(galaxymodel::GalaxyModelMulticomponent(pot, af, dfC),
            point, 1e-3, 10000, &densArrC[0], NULL, NULL, NULL, NULL, NULL);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfO),
            point, 1e-3, 100000, &densOrig, NULL, NULL, NULL, NULL, NULL);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfL),
            point, 1e-3, 100000, &densIntL, NULL, NULL, NULL, NULL, NULL);
        computeMoments(galaxymodel::GalaxyModel(pot, af, dfC),
            point, 1e-3, 100000, &densIntC, NULL, NULL, NULL, NULL, NULL);
        double densSumL=0;
        for(unsigned int i=0; i<densArrL.size(); i++) {
            densSumL += densArrL[i];
            strmL << densArrL[i] << '\n';
        }
        strmL << '\n';
        double densSumC=0;
        for(unsigned int i=0; i<densArrC.size(); i++) {
            densSumC += densArrC[i];
            strmC << densArrC[i] << '\n';
        }
        strmC << '\n';
        strm << r << ' ' << pot.density(point) << '\t' << densOrig << '\t' <<
            densIntL << ' ' << densSumL << '\t' << densIntC << ' ' << densSumC << '\n';
        ok &= math::fcmp(densOrig, densIntL, 5e-2)==0 && math::fcmp(densSumL, densIntL, 2e-2)==0 && 
              math::fcmp(densOrig, densIntC, 5e-2)==0 && math::fcmp(densSumC, densIntC, 2e-2)==0;
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}
