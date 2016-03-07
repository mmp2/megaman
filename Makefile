CURRENT_DIR = $(shell pwd)
TEST_DIR = /tmp/megaman
PKG = megaman

install:
	python setup.py install

clean:
	rm -r build/

test-dir:
	mkdir -p $(TEST_DIR)

test: test-dir install
	cd $(TEST_DIR) && nosetests $(PKG)

doctest: test-dir install
	cd $(TEST_DIR) && nosetests --with-doctest $(PKG)

test-coverage: test-dir install
	cd $(TEST_DIR) && nosetests --with-coverage --cover-package=$(PKG) $(PKG)

test-coverage-html: test-dir install
	cd $(TEST_DIR) && nosetests --with-coverage --cover-html --cover-package=$(PKG) $(PKG)
	rsync -r $(TEST_DIR)/cover $(CURRENT_DIR)/
	echo "open ./cover/index.html with a web browser to see coverage report"
