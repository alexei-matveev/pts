#
# Type "make -k" to run all tests,
# (without -k make will stop at first failure it encounters)
#
# Type "make manual" to create a pdf version of the newest
# sources for the manual in doc subfolder
#
IO = ./io
PES = ./pes
COSOPT = ./cosopt
TOOL = ./tools

src =	\
	common.py \
	sched.py \
	searcher.py \
	ridders.py \
	npz.py \
	func.py \
	ode.py \
	path.py \
        quat.py \
	bezier.py \
	chebyshev.py \
	zmat.py \
	memoize.py \
	bfgs.py \
	fopt.py \
	qfunc.py \
	chain.py \
        steepest_decent.py \
        simple_decent.py \
	rc.py \
        dct.py \
	paramap.py \
	vib.py \
	metric.py \
        threepointmin.py \
	cfunc.py \
	dimer.py \
	srcio \
	srcpes \
	srccosopt \
	srctools \

# dont call it "test" as we have a directory called so:
test-all: $(src:.py=.doctest)

manual :
	$(MAKE) -C ./doc

# run a doctest on the module, return failure if any of the tests fail:
%.doctest: %.py
	python -c "import $*, doctest, sys; errs, _ = doctest.testmod($*); sys.exit(bool(errs))"

srcio:
	$(MAKE) -C $(IO)

srcpes:
	$(MAKE) -C $(PES)

srccosopt:
	$(MAKE) -C $(COSOPT)

srctools:
	$(MAKE) -C $(TOOL)

