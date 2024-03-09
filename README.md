# Pattern Recognition  
Pattern recognition assignment for the 5th-semester course on pattern recognition.
## Run Locally  

Clone the project  

~~~bash  
  git clone https://github.com/Noudi03/PatternRecognition
~~~

Go to the project directory  

~~~bash  
  cd PatternRecognition
~~~

## Recommended: Create a virtual environment  

~~~bash
  pip install virtualenv
~~~

Create a Virtual Environment named venv
~~~bash
  python -m venv venv
~~~

## Activation
Activate the virtual environment
### Windows
~~~bash
  .venv\Scripts\activate
~~~

Note: if you cannot run the activation script, open an elevated powershell and execute the following command
~~~ps1
  set-executionpolicy remotesigned
~~~
### Linux
~~~bash
  source venv/bin/activate
~~~

## Install dependencies  
We need this package in order to build the project with setuptools
~~~bash  
  pip install build
~~~
## Build Package

~~~bash  
  pip install .
~~~

## Run the main script
~~~bash
  python -m src
~~~


## License  

[MIT](LICENSE)
