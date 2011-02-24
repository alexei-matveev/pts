#
# Type "make -k" to run all tests,
# (without -k make will stop at first failure it encounters)
#

src =	\
	common.py \
	sched.py \
	searcher.py \
	ridders.py \
	npz.py \
	func.py \
	ode.py \
	path.py \
	bezier.py \
	chebyshev.py \
	zmat.py \
	mueller_brown.py \
	memoize.py \
	bfgs.py \
	fopt.py \
	qfunc.py \
	chain.py \
	rc.py \
	paramap.py \
	vib.py \
	metric.py \
	cfunc.py \

# dont call it "test" as we have a directory called so:
test-all: $(src:.py=.doctest)

# run a doctest on the module, return failure if any of the tests fail:
%.doctest: %.py
	python -c "import $*, doctest, sys; errs, _ = doctest.testmod($*); sys.exit(bool(errs))"
