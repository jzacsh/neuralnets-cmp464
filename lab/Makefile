UNIT_TT  :=  $(filter-out __init__.py,$(wildcard *_test.py))
LAB_PYS  :=  $(sort $(filter-out __init__.py $(UNIT_TT),$(wildcard *.py)))

current: $(lastword $(LAB_PYS))

all: test labs

test: $(UNIT_TT)

labs: $(LAB_PYS)

$(LAB_PYS) $(UNIT_TT):
	@python3 $@

.PHONY: current labs $(LAB_PYS) $(UNIT_TT)
