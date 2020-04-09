## Multilogue-net - Official PyTorch Implementation  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multilogue-net-a-context-aware-rnn-for-multi/multimodal-sentiment-analysis-on-cmu-mosei)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-cmu-mosei?p=multilogue-net-a-context-aware-rnn-for-multi) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multilogue-net-a-context-aware-rnn-for-multi/multimodal-sentiment-analysis-on-mosi)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-mosi?p=multilogue-net-a-context-aware-rnn-for-multi) 

This repository contains the official implemention for the following paper:
> **Multilogue-Net: A Context Aware RNN for Multi-modal Emotion Detection and Sentiment Analysis in Conversation**<br>
> https://arxiv.org/abs/2002.08267
>
> **Abstract:** *Sentiment Analysis and Emotion Detection in conversation is key in a number of real-world applications, with different applications leveraging different kinds of data to be able to achieve reasonably accurate predictions. Multimodal Emotion Detection and Sentiment Analysis can be particularly useful as applications will be able to use specific subsets of the available modalities, as per their available data, to be able to produce relevant predictions. Current systems dealing with Multimodal functionality fail to leverage and capture the context of the conversation through all modalities, the current speaker and listener(s) in the conversation, and the relevance and relationship between the available modalities through an adequate fusion mechanism. In this paper, we propose a recurrent neural network architecture that attempts to take into account all the mentioned drawbacks, and keeps track of the context of the conversation, interlocutor states, and the emotions conveyed by the speakers in the conversation. Our proposed model out performs the state of the art on two benchmark datasets on a variety of accuracy and regression metrics.*  
  
| ![dialogue](https://github.com/amanshenoy/multilogue-net/blob/master/diagrams/dialogue.jpg) |
|:-------------------------------------------------------------------------------------------:|
| Multilogue-net architecture for updates and classification for two timestamps in dialogue |

## Resources and Dependancies

The datasets used to train all the models were obtained and preprocessed using the CMU-Multimodal SDK which can be found [here](https://github.com/A2Zadeh/CMU-MultimodalSDK).  

The `data` folder in the repository contains pre-processed data for the CMU-MOSEI Dataset, whose details can be found [here](https://www.aclweb.org/anthology/P18-1208/).

The repository contains files consisting of all relevant models, dataloaders, formatted data, and training scripts to be able to train the model.  

The models in the repositories can be trained on the following target variables -  

* Binary Sentiment labels  
* Emotion labels (One of 6 emotions)
* Regression outputs (Real valued range between -3 to +3)  
  
The repository also contains a `.txt` requirements file consisting of all dependancies required to be able to train and infer the model on any of the labels by running

    >> pip install -r requirements.txt

## Implementation and training

The repository contains three training scripts as per the desired target variables.  

The scripts require python3.6 or above and can be run as

    >> python train_categorical --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --log_dir='logs/mosei_categorical'
  
    >> python train_emotion --no-cuda=False --lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 --batch-size=128 --epochs=50 --class-weight=True --emotion='happiness' --log_dir='logs/mosei_emotion'
  
    >> python train_regression --no-cuda=False --lr=1e-4 --l2=1e-4 --rec-dropout=0.1 --dropout=0.25 --batch-size=128 --epochs=100 --log_dir='logs/mosei_regression'
    
Depending on the kind of prediction desired.

The model can further be extended onto other datasets for any number of parties in the conversation. The dataloader stores the data for each example in a particular format. The dictionary keys for any example would be -    

    [ID, speakers, labels, text_feat, audio_feat, visual_feat, sentence_in_text, train, test]
    
where,
- ID is the identification number for the video
- speakers is a list consisting of a label indicating which one of the speakers spoke the corresponding utterance (for example - If conversation alternated between A and B for a total of 4 utterances, the list would be ['A', 'B', 'A', 'B'])
- labels are the corresponding labels for that utterance
- text_feat are text features (GLoVe embeddings in our case)
- audio_feat are audio features (OpenSMILE features in our case)
- video_feat are video features (FACET 2.0 in our case)
- sentence_in_text is the sentence corresponding to an utterance
- train and test are lists of the ID's of all the examples belonging to the train and test set respectively

## Experimentation and Results 

The model takes roughly 15 seconds/epoch for `train_emotion.py` and `train_categorical.py` and 40 seconds/epoch for `train_regression.py` on CMU-MOSEI, on a single NVIDIA GV100 and achieves state-of-the-art performance (at the time of writing) on emotion recognition, binary sentiment prediction, and sentiment regression problems.
  
| ![table](https://github.com/amanshenoy/multilogue-net/blob/master/diagrams/emotion-results.jpg) |
|:-----------------------------------------------------------------------------------------------:|
| Results of Multilogue-net on emotion labels on CMU-MOSEI dataset |


