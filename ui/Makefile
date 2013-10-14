#
# Type "make -k" to run all tests,
# (without -k make will stop at first failure it encounters)
#

srcio =   \
	read_inputs.py \
	read_COS.py \


# dont call it "test" as we have a directory called so:
#test-all: $(src:.py=.doctest) $(srcio:.py=.doctestio)
test-all: $(srcio:.py=.doctest)

# run a doctest on the module, return failure if any of the tests fail:
%.doctest: %.py
	python -c "import $*, doctest, sys; errs, _ = doctest.testmod($*); sys.exit(bool(errs))"

