test:
	python -m pytest

build:
	tsc -p lake_monster_js
	cp lake_monster_js/src/index.html lake_monster_js/dist
	cp lake_monster_js/src/styles.css lake_monster_js/dist

deploy:
	sudo gh-pages -d dist

localhost:
	python -m http.server --directory lake_monster_js/dist/


convert:
	tensorflowjs_converter \
		--input_format=tf_saved_model \
		--output_format=tfjs_graph_model \
		data/temp/saved_model lake_monster_js/dist/saved_model