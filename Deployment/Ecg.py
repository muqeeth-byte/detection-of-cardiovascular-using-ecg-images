from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu,gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# VERSION WITHOUT NATSORT DEPENDENCY
# ============================================

class ECG:
	def  getImage(self,image):
		"""
		this functions gets user image
		return: user image
		"""
		image=imread(image)
		return image

	def GrayImgae(self,image):
		"""
		This funciton converts the user image to Gray Scale
		return: Gray scale Image
		"""
		image_gray = color.rgb2gray(image)
		image_gray=resize(image_gray,(1572,2213))
		return image_gray

	def DividingLeads(self,image):
		"""
		This Funciton Divides the Ecg image into 13 Leads including long lead. Bipolar limb leads(Leads1,2,3). Augmented unipolar limb leads(aVR,aVF,aVL). Unipolar (+) chest leads(V1,V2,V3,V4,V5,V6)
  		return : List containing all 13 leads divided
		"""
		Lead_1 = image[300:600, 150:643] # Lead 1
		Lead_2 = image[300:600, 646:1135] # Lead aVR
		Lead_3 = image[300:600, 1140:1625] # Lead V1
		Lead_4 = image[300:600, 1630:2125] # Lead V4
		Lead_5 = image[600:900, 150:643] #Lead 2
		Lead_6 = image[600:900, 646:1135] # Lead aVL
		Lead_7 = image[600:900, 1140:1625] # Lead V2
		Lead_8 = image[600:900, 1630:2125] #Lead V5
		Lead_9 = image[900:1200, 150:643] # Lead 3
		Lead_10 = image[900:1200, 646:1135] # Lead aVF
		Lead_11 = image[900:1200, 1140:1625] # Lead V3
		Lead_12 = image[900:1200, 1630:2125] # Lead V6
		Lead_13 = image[1250:1480, 150:2125] # Long Lead

		#All Leads in a list
		Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
		fig , ax = plt.subplots(4,3)
		fig.set_size_inches(10, 10)
		x_counter=0
		y_counter=0

		#Create 12 Lead plot using Matplotlib subplot

		for x,y in enumerate(Leads[:len(Leads)-1]):
			if (x+1)%3==0:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				y_counter+=1
	    
		#save the image
		fig.savefig('Leads_1-12_figure.png')
		fig1 , ax1 = plt.subplots()
		fig1.set_size_inches(10, 10)
		ax1.imshow(Lead_13)
		ax1.set_title("Leads 13")
		ax1.axis('off')
		fig1.savefig('Long_Lead_13_figure.png')

		return Leads

	def PreprocessingLeads(self,Leads):
		"""
		This Function Performs preprocessing to on the extracted leads.
		"""
		fig2 , ax2 = plt.subplots(4,3)
		fig2.set_size_inches(10, 10)
		#setting counter for plotting based on value
		x_counter=0
		y_counter=0

		for x,y in enumerate(Leads[:len(Leads)-1]):
			#converting to gray scale
			grayscale = color.rgb2gray(y)
			#smoothing image
			blurred_image = gaussian(grayscale, sigma=1)
			#thresholding to distinguish foreground and background
			#using otsu thresholding for getting threshold value
			global_thresh = threshold_otsu(blurred_image)

			#creating binary image based on threshold
			binary_global = blurred_image < global_thresh
			#resize image
			binary_global = resize(binary_global, (300, 450))
			if (x+1)%3==0:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				y_counter+=1
		fig2.savefig('Preprossed_Leads_1-12_figure.png')

		#plotting lead 13
		fig3 , ax3 = plt.subplots()
		fig3.set_size_inches(10, 10)
		#converting to gray scale
		grayscale = color.rgb2gray(Leads[-1])
		#smoothing image
		blurred_image = gaussian(grayscale, sigma=1)
		#thresholding to distinguish foreground and background
		#using otsu thresholding for getting threshold value
		global_thresh = threshold_otsu(blurred_image)
		print(global_thresh)
		#creating binary image based on threshold
		binary_global = blurred_image < global_thresh
		ax3.imshow(binary_global,cmap='gray')
		ax3.set_title("Leads 13")
		ax3.axis('off')
		fig3.savefig('Preprossed_Leads_13_figure.png')


	def SignalExtraction_Scaling(self,Leads):
		"""
		This Function Performs Signal Extraction using various steps,techniques: conver to grayscale, apply gaussian filter, thresholding, perform contouring to extract signal image and then save the image as 1D signal
		"""
		fig4 , ax4 = plt.subplots(4,3)
		#fig4.set_size_inches(10, 10)
		x_counter=0
		y_counter=0
		for x,y in enumerate(Leads[:len(Leads)-1]):
			#converting to gray scale
			grayscale = color.rgb2gray(y)
			#smoothing image
			blurred_image = gaussian(grayscale, sigma=0.7)
			#thresholding to distinguish foreground and background
			#using otsu thresholding for getting threshold value
			global_thresh = threshold_otsu(blurred_image)

			#creating binary image based on threshold
			binary_global = blurred_image < global_thresh
			#resize image
			binary_global = resize(binary_global, (300, 450))
			#finding contours
			contours = measure.find_contours(binary_global,0.8)
			contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
			for contour in contours:
				if contour.shape in contours_shape:
					test = resize(contour, (255, 2))
			if (x+1)%3==0:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				y_counter+=1
	    
			#scaling the data and testing
			lead_no=x
			scaler = MinMaxScaler()
			fit_transform_data = scaler.fit_transform(test)
			Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
			Normalized_Scaled=Normalized_Scaled.T
			#scaled_data to CSV
			if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
			else:
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
	      
		fig4.savefig('Contour_Leads_1-12_figure.png')


	def CombineConvert1Dsignal(self):
		"""
		This function combines all 1D signals of 12 Leads into one File csv for model input.
		returns the final dataframe - WITHOUT NATSORT
		"""
		# Get list of CSV files (without natsort)
		location = os.getcwd()
		csv_files = []
		
		for file in os.listdir(location):
			if file.startswith('Scaled_1DLead_') and file.endswith('.csv'):
				csv_files.append(file)
		
		# Sort naturally without natsort
		# Extract lead numbers and sort
		csv_files_with_nums = []
		for file in csv_files:
			try:
				# Extract number from filename: Scaled_1DLead_1.csv → 1
				num = int(file.replace('Scaled_1DLead_', '').replace('.csv', ''))
				csv_files_with_nums.append((num, file))
			except:
				pass
		
		# Sort by lead number
		csv_files_with_nums.sort(key=lambda x: x[0])
		csv_files = [f[1] for f in csv_files_with_nums]
		
		print(f"✓ Found {len(csv_files)} lead files: {csv_files}")
		
		# Read first file
		if csv_files:
			test_final = pd.read_csv(csv_files[0])
			
			# Combine remaining files
			for file in csv_files[1:]:
				try:
					df = pd.read_csv(file)
					test_final = pd.concat([test_final, df], axis=1, ignore_index=True)
				except Exception as e:
					print(f"Warning: Could not read {file}: {e}")
			
			return test_final
		else:
			print("Error: No CSV files found!")
			return None

	
	def DimensionalReduciton(self,test_final):
		"""
		This function reduces the dimensionality of the 1D signal using PCA
		returns the final dataframe
		"""
		try:
			# Try to load the trained PCA model
			pca_loaded_model = joblib.load('PCA_ECG (1).pkl')
		except Exception as e:
			print(f"Warning: Could not load PCA model: {e}")
			print("Using fallback PCA with n_components=400")
			# Fallback: Create new PCA if loading fails
			pca_loaded_model = PCA(n_components=min(400, test_final.shape[1]))
			pca_loaded_model.fit(test_final)
		
		try:
			result = pca_loaded_model.transform(test_final)
			final_df = pd.DataFrame(result)
			return final_df
		except Exception as e:
			print(f"Error during PCA transformation: {e}")
			# If transformation fails, return original data
			return test_final

	def ModelLoad_predict(self, final_df):
		"""
		ULTRA-ROBUST VERSION: Handles scikit-learn version compatibility
		Tries multiple approaches to load and predict
		"""
		import pickle
		import sys
		
		diagnosis_map = {
			0: "Your ECG corresponds to Abnormal Heartbeat",
			1: "Your ECG corresponds to Myocardial Infarction",
			2: "Your ECG is Normal",
			3: "Your ECG corresponds to History of Myocardial Infarction"
		}
		
		model_file = 'Heart_Disease_Prediction_using_ECG (4).pkl'
		
		# ========== ATTEMPT 1: Try standard joblib loading ==========
		try:
			print("[Attempt 1] Loading model with joblib...")
			loaded_model = joblib.load(model_file)
			print("✓ Model loaded successfully with joblib")
			result = loaded_model.predict(final_df)
			return diagnosis_map.get(int(result[0]), f"Unknown diagnosis (Class {result[0]})")
		except (ValueError, TypeError) as e:
			print(f"✗ joblib loading failed: {str(e)[:100]}")
		except Exception as e:
			print(f"✗ joblib loading error: {str(e)[:100]}")
		
		# ========== ATTEMPT 2: Try pickle.load instead ==========
		try:
			print("[Attempt 2] Loading model with pickle.load...")
			with open(model_file, 'rb') as f:
				loaded_model = pickle.load(f)
			print("✓ Model loaded successfully with pickle")
			result = loaded_model.predict(final_df)
			return diagnosis_map.get(int(result[0]), f"Unknown diagnosis (Class {result[0]})")
		except (ValueError, TypeError) as e:
			print(f"✗ pickle loading failed: {str(e)[:100]}")
		except Exception as e:
			print(f"✗ pickle loading error: {str(e)[:100]}")
		
		# ========== ATTEMPT 3: Try rebuilding the model structure ==========
		try:
			print("[Attempt 3] Attempting to fix model compatibility...")
			import warnings
			warnings.filterwarnings('ignore')
			
			# Load with unsafe mode (handles some compatibility issues)
			with open(model_file, 'rb') as f:
				import importlib
				# Try to load with compatibility fixes
				loaded_model = pickle.load(f, fix_imports=True, encoding='latin1')
			
			print("✓ Model loaded with compatibility fixes")
			result = loaded_model.predict(final_df)
			return diagnosis_map.get(int(result[0]), f"Unknown diagnosis (Class {result[0]})")
		except Exception as e:
			print(f"✗ Compatibility fix failed: {str(e)[:100]}")
		
		# ========== ATTEMPT 4: Return helpful error message ==========
		print("\n" + "="*70)
		print("CRITICAL ERROR: Cannot load pre-trained model")
		print("="*70)
		print("\nSolution: Run the retraining script")
		print("Command: python retrain_model_standalone.py")
		print("="*70 + "\n")
		
		return "❌ CRITICAL: Model Loading Failed. Run: python retrain_model_standalone.py"
