# Face_Recognition

Dataset = Data consists in Data Folder which consists of 6 categories in which each category consist of 21 images

Test Set =  Test Images are present in Test_Images Folder

Python Files = Total 6 python files

    1. encode_faces.py
    2. recognize_faces_image.py
    3. recognize_faces_video.py
    4. Adding_new_dataset.py --optional
    5. change_file_names.py --optional

We have used face_recognition library which was in built in python to detect faces in images 

--face_recognition library is trained with 3 million images to detect 128 features of faces which is given in 
128 d row vector

--face_recognition library has a variation of kNN classifier which identifies whether a face is nearby to other 
not by invoking face_recognition.compare_faces()

--We give all the images in the dataset to face_recognition.face_encoder() method to get 128 d vector and then
store them in Dictionary = {"encodings": calculated encodings (list type), "names": names (list type)}

--When a face is given as an input then 128 d vector is calculated and we calculate the Euclidean Distance between 
two vectors 

--Likewise the query vector is compared with all the vectors present in the dictionary that is present in
encoding.pickle files

--I created Adding_new_dataset.py file to just append new calculated vectors and names in the dictionary rather than 
encoding it from first of the dataset

--In face_recognition library there are two methods of detecting faces one is through calculating "HOG" vectors or
through "CNN" but if we use CPU then to minimise the computing time we have choose "HOG" but it gives less accurate 
results when compared to "CNN". If we use GPU then using "CNN" gives good results. 
