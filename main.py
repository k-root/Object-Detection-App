
import flaskFiles

import os

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./springmldemoproject.json"

app = flaskFiles.create_app()


# This is only used when running locally. When running live, gunicorn runs
# the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)