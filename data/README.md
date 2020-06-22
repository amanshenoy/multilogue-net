## Generating data for re-training or inference

All datasets used in the paper and in this repository is processed data from the [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

The model can further be extended onto other datasets for training on any number of parties in the conversation and also for inference. 

The dataloader stores the data for each example in a particular format. The dictionary keys for any example would be -    

    [ID, speakers, labels, text_feat, audio_feat, visual_feat, sentence_in_text, train, test]
    
where each dictionary key is described as follows:    
- ID is the identification number for the video
- speakers is a list consisting of a label indicating which one of the speakers spoke the corresponding utterance (for example - If conversation alternated between A and B for a total of 4 utterances, the list would be `['A', 'B', 'A', 'B']`)
- labels are the corresponding labels for that utterance
- text_feat are text features (GLoVe embeddings in our case)
- audio_feat are audio features (OpenSMILE features in our case)
- video_feat are video features (FACET 2.0 in our case)
- sentence_in_text is the sentence corresponding to an utterance
- train and test are lists of the ID's of all the examples belonging to the train and test set respectively
