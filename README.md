# What this repo about
This code base is to write all the code needed for education purposes and to run small experiments to learn about different topics in alignment

# Steps to run the code by setting up venv
- Create venv using `python -m venv .env`
- Activate venv using `.env\Scripts\activate.bat`
- Run `curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py`
- Run `python get-pip.py --force-reinstall`
- Install pacakges using `pip install -r requirements.txt`
- To freeze the requirments.txt file `pip freeze > requirements.txt`
- Run this command if you get the package error `pip install urllib3==1.26.6`
- Run the backend `python manage.py runserver 5000`
