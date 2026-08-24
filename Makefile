.PHONY: install train evaluate test serve web clean

install:
	pip install -r requirements.txt --break-system-packages

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest -v

serve:
	python -m mcp_server.server

web:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
