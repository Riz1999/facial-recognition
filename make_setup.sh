python3 -m pip install virtualenv
python3 -m virtualenv env
env\Scripts\activate
echo "Environment activated"
echo "Installing requirements"
python3 -m pip install -r requirements.txt
echo "Requirements installed"