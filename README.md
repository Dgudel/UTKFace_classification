# UTKFace_classification

The project uses the human face images' dataset uploaded from the Kaggle website (https://www.kaggle.com/datasets/jangedoo/utkface-news). UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 104 years old). The dataset consists of over 20000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.

In total, there are 57596 image files in five folders: 'part1', 'part2', 'part3', 'UTKFace' and 'crop_part1'. Some of these images are duplicates. Also, the folders contain images of diferent size and resolution representing the the same persons.Also, the dataset provides the corresponding landmarks (68 points). Images are labelled by age, gender, and ethnicity. the label info is embedded in the names of image files, formated like [age][gender][race]_[date&time].jpg

- [age] is an integer from 0 to 116, indicating the age
- [gender] is either 0 (male) or 1 (female)
- [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

The purpose of the project is to apply transfer learning for training multitask classifiers which would be able to corectly predict age and gender for the face images - that is, to provide gender and age labels to images of faces of various persons.

The project files consist of:

- the 'UTKFace.ipynb' file which provides the the exploratory analysis of the dataset of face images and the report of building, training, fine-tuning and evaluating classifiers able to predict age and gender of face images as well as discussing ethical issues of the model applicability.
- the 'utkface_dataset.py' file which includes classes used for the importing and preprocessing of images.
