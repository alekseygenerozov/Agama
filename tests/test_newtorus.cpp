/** \file    test_newtorus.cpp
    \author  Eugene Vasiliev
    \date    February 2016

*/
#include "potential_perfect_ellipsoid.h"
//#include "actions_staeckel.h"
//#include "math_core.h"
#include "actions_newtorus.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

const double axis_a=1.6, axis_c=1.0; // axes of perfect ellipsoid

bool test_torus(const potential::OblatePerfectEllipsoid& pot, const actions::Actions acts)
{
    actions::ActionMapperNewTorus tor(pot, acts);
    return true;
}

int main()
{
    const potential::OblatePerfectEllipsoid potential(1.0, axis_a, axis_c);
    bool allok=true;
    allok &= test_torus(potential, actions::Actions(1, 1, 1));
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}