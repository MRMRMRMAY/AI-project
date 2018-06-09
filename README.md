# AI-project
Python Edition: 3.6.5

Please open cmd and move to the project folder, then type the command "pip install -r requirements.txt".

Please download and decompress the following files from the  "Data.zip" into the folder¡°data".
	
	"Gudi_model_10_epochs_2000_faces.data-00000-of-00001"
	"Gudi_model_10_epochs_2000_faces.index"
	"Gudi_model_10_epochs_2000_faces.meta"
	"fer2013.csv"

The data is in CSV and we need to transform it using the script "csv_to_numpy.py" that generates the image and label data in the data folder.
 
Please install "AVbin10-win64".

#Usage :
	
	To train a model, using "emotion_recognition.py" . The command is 'python3 emotion_recognition.py train".
	
	To use our application, execute  "smartbuddy.py".
	
