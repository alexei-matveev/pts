#
# Type "make -k" to run all tests,
# (without -k make will stop at first failure it encounters)
#

src =	\
	calcman.py \
	common.py \
	coord_sys.py \
	sched.py \
	searcher.py \
	ridders.py \
	func.py \
	path.py \
	zmat.py \
	mueller_brown.py \
	memoize.py \
	bfgs.py \
	fopt.py \
	qfunc.py \
	chain.py \

# dont call it "test" as we have a directory called so:
test-all: $(src:.py=.doctest)

# run a doctest on the module, return failure if any of the tests fail:
%.doctest: %.py
	python -c "import $*, doctest, sys; errs, _ = doctest.testmod($*); sys.exit(bool(errs))"
