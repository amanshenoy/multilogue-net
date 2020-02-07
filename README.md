## Multilogue-net
This repository contains the official pytorch implemention for the paper [Multilogue-Net: An inter-modal attentive reccurent neural network architecture for multimodal sentiment analysis and emotion recognition](amanshenoy.github.io). The reposiory contains a model file, a dataloader file and, training scripts and data to be able to train the model on the [CMU-MOSEI Dataset](https://www.aclweb.org/anthology/P18-1208/) for -   

* Binary Sentiment labels  
* Emotion labels (One of 6 emotions)
* Regression outputs (Real range between -3 to +3)  
  
The repository also contains a `.txt` requirements file consisting of all dependancies required to be able to train and infer the model on any of the labels by running

    >> pip install -r requirements.txt

The model captures context and speaker states by monitoring three sequential representations and finally fusing the representations from all modalities using a pairwise fusion mechanism  
![dialogue](https://github.com/amanshenoy/multilogue-net/blob/master/diagrams/dialogue.jpg)

Further model details regarding training, inference and architecture can be found in the paper linked above

## Implementation and training

The scripts require python3.6 or above and can be run as

    >> python train_categorical --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --log_dir='logs/mosei_categorical'
  
    >> python train_emotion --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --emotion='happiness' --log_dir='logs/mosei_emotion'
  
    >> python train_regression --no-cuda=False --lr=1e-4 --l2=1e-4 --rec-dropout=0.1 --dropout=0.25 --batch-size=128 --epochs=100 --log_dir='logs/mosei_regression'
    
Depending on the kind of prediction desired.

The CMU-MOSEI dataset has single party conversation data from YouTube of 2199 opinion video clips with labels available for sentiment within the real range -3 to 3, and for emotion labels for each utterance. The binary sentiment labels are obtained by considering sentiment >= 0 to be 1 and all others to be 0.   

The model can further be extended onto other datasets for any number of parties in the conversation. The dataloader stores the data for each example in a particular format.   
The dictionary keys for any example would be -  

    [ID, speakers, labels, text_feat, audio_feat, visual_feat, sentence_in_text, train, test]
    
where,
* ID is the identification number for the video
* speakers is a list consisting of a label indicating which one of the speakers spoke the corresponding utterance (example : If conversation alternated between A and B for a total of 4 utterances, the list would be ['A', 'B', 'A', 'B'])
* labels are the corresponding labels for that utterance
* text_feat are text features (GLoVe embeddings in our case)
* audio_feat are audio features (OpenSMILE features in our case)
* video_feat are video features (FACET 2.0 in our case)
* sentence_in_text is the sentence corresponding to an utterance
* train and test are lists of the ID's of all the examples belonging to the train and test set respectively

## Concluding 

The model takes roughly 15 seconds/epoch for `train_emotion.py` and `train_categorical.py` and 40 seconds/epoch for `train_regression.py` on CMU-MOSEI, on a single NVIDIA GV100 and achieves state-of-the-art performance (at the time of writing) on emotion recognition, binary sentiment prediction, and sentiment regression problems.

The following work done as a part of an internship project at NVIDIA Graphics, Bengaluru

