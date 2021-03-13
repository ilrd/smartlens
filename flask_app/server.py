from flask import Flask, request, jsonify, render_template
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'src/')

from src.scraping import parse_page
from src.model import get_images_category

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/')
def root_get():
    return render_template('categorize_profile.html')


@app.route('/', methods=['POST'])
def root_post():
    try:
        url = request.get_json(force=True)['url']

        scraped_images = parse_page(url)
        best_category = get_images_category(scraped_images)

        response = jsonify({
            'predicted_category': str(best_category),
            'img_original_path': 'static/best_original.jpg',
            'img_result_path': 'static/best_result.jpg',
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    except:
        pass


if __name__ == "__main__":
    print("Website is up!")
    from waitress import serve

    serve(app, host="localhost", port=5000)
