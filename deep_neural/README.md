This is the readme file for Jae Han's Final Project Submission: CNN

The dnn.py file is the main script used to create and train the networks, this does not need to be run by the grader
and is merely a proof of work/concept.

transfer_learning.py is an extenuation of the dnn.py file, which imports a keras VGG16 model that is pretrained,
so that I can apply transfer learning techniques for the SVHN dataset.

load_models.py is a helper function for run_v3.py which prints a summary, configuration and score for each model.

the link to my videos is: https://www.youtube.com/watch?v=cGVp3u6B6jY&feature=youtu.be

##########################################################################################################################

run_v3.py:
This is the main file that the autograder will use to run the code, and view the output of the predictions given the images
we found in the graded_images directory. The grader WILL NOT need to call any functions, and only needs to run this script.

The console will report all necessary information

This script will ask the grader a [yes, no] question

Q: Do you want to view the models, summaries, and configurations? There are 5 models and might take some time.
    a. if the grader enters 'yes' then the models will be loaded and viewable via the console.
    b. if the grader says 'no'
        the final images (5 images) will be pulled from the graded_images dir, and tested against each of the 5 models
        with the prediction values being returned in the console

NOTE: It is NOT necessary to view the models (step a.) if the grader needs to see the configuration. Each model has it's own
pickled configuration file if details are necessary. The load_models.py helper script will just be used to display these configurations
after loading each model
    

