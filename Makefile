# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

SRCDIR    = src
OBJDIR    = obj
LIBDIR    = lib
EXEDIR    = exe
TESTSDIR  = tests
TORUSDIR  = src/torus

# sources of the main library
SOURCES   = \
            actions_genfnc.cpp \
            actions_interfocal_distance_finder.cpp \
            actions_isochrone.cpp \
            actions_newtorus.cpp \
            actions_spherical.cpp \
            actions_staeckel.cpp \
            actions_torus.cpp \
            coord.cpp \
            cubature.cpp \
            df_base.cpp \
            df_disk.cpp \
            df_factory.cpp \
            df_halo.cpp \
            df_interpolated.cpp \
            galaxymodel.cpp \
            galaxymodel_selfconsistent.cpp \
            galaxymodel_spherical.cpp \
            math_core.cpp \
            math_fit.cpp \
            math_linalg.cpp \
            math_ode.cpp \
            math_optimization.cpp \
            math_sample.cpp \
            math_specfunc.cpp \
            math_sphharm.cpp \
            math_spline.cpp \
            orbit.cpp \
            particles_io.cpp \
            potential_analytic.cpp \
            potential_base.cpp \
            potential_composite.cpp \
            potential_cylspline.cpp \
            potential_dehnen.cpp \
            potential_factory.cpp \
            potential_ferrers.cpp \
            potential_galpot.cpp \
            potential_multipole.cpp \
            potential_perfect_ellipsoid.cpp \
            potential_sphharm.cpp \
            potential_utils.cpp \
            utils.cpp \
            utils_config.cpp \
            fortran_wrapper.cpp \
            py_wrapper.cpp

# ancient Torus code
TORUSSRC  = CHB.cc \
            Fit.cc \
            Fit2.cc \
            GeneratingFunction.cc \
            Orb.cc \
            PJMCoords.cc \
            PJMNum.cc \
            Point_ClosedOrbitCheby.cc \
            Point_None.cc \
            Torus.cc \
            Toy_Isochrone.cc \
            WD_Numerics.cc

# test and example programs
TESTSRCS  = test_math_core.cpp \
            test_math_linalg.cpp \
            test_math_spline.cpp \
            test_coord.cpp \
            test_units.cpp \
            test_orbit_integr.cpp \
            test_potentials.cpp \
            test_potential_expansions.cpp \
            test_isochrone.cpp \
            test_staeckel.cpp \
            test_action_finder.cpp \
            test_torus.cpp \
            test_df_halo.cpp \
            test_df_interpolated.cpp \
            test_df_spherical.cpp \
            example_actions_nbody.cpp \
            example_df_fit.cpp \
            example_fokker_planck.cpp \
            example_self_consistent_model.cpp \

TESTFORTRAN = example_fortran.f

LIBNAME_A  = $(LIBDIR)/libagama.a
LIBNAME_SO = $(LIBDIR)/agama.so
ABSLIBNAME = $(abspath $(LIBNAME_SO))
OBJECTS    = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
TESTEXE    = $(patsubst %.cpp,$(EXEDIR)/%.exe,$(TESTSRCS))
TORUSOBJ   = $(patsubst %.cc,$(OBJDIR)/%.o,$(TORUSSRC))
TESTEXEFORTRAN = $(patsubst %.f,$(EXEDIR)/%.exe,$(TESTFORTRAN))

# autogenerated dependency lists for each cpp file (doesn't work properly yet)
#DEPENDS    = $(OBJDIR)/.depends

all:  $(LIBNAME_SO) $(LIBNAME_A) $(TESTEXE) $(TESTEXEFORTRAN)

#$(DEPENDS):  $(patsubst %.cpp,$(SRCDIR)/%.cpp,$(SOURCES))
#	rm -f $(DEPENDS)
#	$(CXX) $(CXXFLAGS) -MM $^ >> $(DEPENDS)
#include $(DEPENDS)

$(LIBNAME_SO):  $(OBJECTS) $(TORUSOBJ) Makefile Makefile.local
	@mkdir -p $(LIBDIR)
	$(LINK) -shared -o $(ABSLIBNAME) $(OBJECTS) $(TORUSOBJ) $(LFLAGS) $(LIBS)

$(LIBNAME_A):  $(OBJECTS) $(TORUSOBJ) Makefile Makefile.local
	@mkdir -p $(LIBDIR)
	ar rv $(LIBNAME_A) $(OBJECTS) $(TORUSOBJ)

$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME_SO)
	@mkdir -p $(EXEDIR)
	$(LINK) -o "$@" "$<" $(CXXFLAGS) $(ABSLIBNAME) $(LFLAGS)

$(TESTEXEFORTRAN):  $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME_SO)
	$(FC) -o "$@" $(TESTSDIR)/$(TESTFORTRAN) $(ABSLIBNAME) $(LFLAGS) -lstdc++

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp Makefile.local
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) $(DEFINES) $(INCLUDES) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc Makefile.local
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o "$@" "$<"

clean:
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d $(EXEDIR)/*.exe $(LIBNAME)

test:
	cp $(TESTSDIR)/test_all.pl $(EXEDIR)/
	(cd $(EXEDIR); ./test_all.pl)

# if NEMO is present, one may compile the plugin for using external potential within NEMO
ifdef NEMO
NEMOACC = $(NEMOOBJ)/acc/agama.so
nemo:   $(LIBNAME)
	$(CXX) $(CXXFLAGS) -I$(NEMOINC) src/nemo_wrapper.cpp \
	-shared -o $(NEMOACC) $(OBJECTS) $(LFLAGS) $(LIBS) -lnemo
endif

.PHONY: clean test
