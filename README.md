
# Detecting Neurodegenerative Diseases through Speech: A Machine Learning Approach  / AI Builders…

Thai Pseudobulbar Palsy Detection with Wav2Vec and OpenAI Whisper — Jirat Chiaranaipanich

## Introduction / Problem Statement

Neurodegenerative diseases affect 1 in 6 people—nearly 1 billion people worldwide. This group of illnesses includes Alzheimer’s, Parkinson’s, and multiple sclerosis, among others. Although each disease has its unique features and clinical manifestations, almost all neurodegenerative diseases result in speech degradation.

Speech and language changes are often the earliest signs of these neurological conditions. Early detection of these symptoms can be crucial in effective medical intervention, yet current diagnostic methods are not accessible to a large portion of people with these diseases. Many diagnostic methods are expensive and can only be conducted in hospitals in large cities with specialized equipment, causing accessibility of neurodegenerative disease early detection to be low amongst less privileged individuals. Machine learning promises a non-invasive, cheap, and accessible method of detecting early signs of neurodegenerative disease onset.

One of the conditions caused by neurodegenerative diseases like Parkinsons is pseudobulbar palsy (PP), where muscles in the corticobulbar tracts are weakened due to trauma, neurodegenerative diseases, or stroke. In this case, the data comes from patients who have PP due to stroke. In this project, I use Wav2Vec and OpenAI Whisper to predict if someone has pseudobulbar palsy or not. The same training scripts could be used to classify other neurodegenerative diseases with speech degradation symptoms.

## Metrics and Baselines

The metrics for this project were provided by the Scikit-Learn library, which calculates the recall, precision, support, and F1 score for the model. Accuracy calculation was self-implemented. Accuracy is simply the number of correct predictions divided by the number of total predictions. Recall, also known as the true positive rate, measures the percent of true positives in all files are actually positive (true positive + false negative.) Precision is the percent of true positives within all files that are classified as positive (true positive + false positive.) F1-score is the harmonic mean of recall and precision. Support is the number of true predictions for each label.

For our baselines, a study on [Mandarin audio detection of Parkinsons ](https://www.frontiersin.org/articles/10.3389/fnagi.2022.1036588/full)achieved an accuracy of ~75%. A study on [English audio detection of Parkinsons](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8564663/) achieved an AUC of ~0.75. There are no baselines for Thai language Parkinsons detection and no baselines for Thai Pseudobulbar Palsy detection.

## Data Collection and Cleaning

Data for this project was graciously provided by Chulalongkorn University. The dataset consists of .wav file voice recordings of participants with PP and without PP (control group.) The dataset was already cleaned and pre-processed and was separated into folders for PP and CT. The folders were further separated into folders for each individual participant, with each folder containing 8 different Thai sentences or phrases:

1. ชาวไร่ตัดต้นสนทำท่อนซุง

1. ปูม้าวิ่งไปมาบนใบไม้ (เน้นใช้ริมฝีปาก)

1. อีกาคอยคาบงูคาบไก่ (เน้นใช้เพดานปาก)

1. เพียงแค่ฝนตกลงที่หน้าต่างในบางครา

1. “อาาาาาาาาาาา”

1. “อีีีีีีีีี”

1. “อาาาา” (ดังขึ้นเรื่อยๆ)

1. “อาา อาาา อาาาาา”

Note: A few individuals were missing up to 3 sentences.

## Exploratory Data Analysis

Before modeling, I listened to many audio files from both the PP and CT folders. Overall, the main distinction between PP and CT was that people with PP tended to have a shaky, slow tone as well as a more hoarse voice. Additionally, some individuals with PP pronounced certain syllables unclearly. These changes in speech are likely due to the neurodegenerative disease’s effects on the muscles that control speech production, which is called dysarthria.

On the other hand, the participants in the control group pronounced the sentences normally without shaky fluctuations.

Try listening to one example of someone with SP: [https://drive.google.com/file/d/1IP836b7V4B4nw9WfMnn9ZMA3E6aggEVL](https://drive.google.com/file/d/1IP836b7V4B4nw9WfMnn9ZMA3E6aggEVL)

And one example of someone without PP (control): [https://drive.google.com/file/d/11OcYfMsrF851dmZACPgYvMK_M0GFkVzg/view?usp=sharing](https://drive.google.com/file/d/11OcYfMsrF851dmZACPgYvMK_M0GFkVzg/view?usp=sharing)

Exploratory data analysis later informed my decision to not use TimeShift as a part of the audio augmentations as some of the speech deformities associated with PP could be replicated by time shifting control audio files. Time-stretching a control audio file may confuse the model as individuals with PP tended to speak slower and more hoarsely, making the audio files less distinguishable from each other.

## Modeling, Validation, and Error Analysis

I tested two pre-trained models, Wav2Vec and OpenAI Whisper. I trained wav2vec with and without audio augmentations and OpenAI Whisper without audio augmentations due to time limits. All models had peak accuracy at above 90%.

![Model pipeline](https://cdn-images-1.medium.com/max/3688/1*aDx5e4ZY0dxVlwKd2ci-2w.png)*Model pipeline*

The training scripts can be viewed at my github page: [https://github.com/spycoderyt/thaidysarthriadetection](https://github.com/spycoderyt/thaidysarthriadetection)

### **Training Process**

The .wav files were extracted from google drive and separated using sci-kit learn train_test_split(). Each group was loaded into a custom dataset and then finally loaded into a DataLoader with batch size 16.

The models were then loaded (facebook/wav2vec2-base-960h and openai/whisper-tiny) for binary classification. Cross entropy loss was used for the loss function, and Adam optimizer was used. A scheduler was also uesd with step_size 5 and gamma = 0.1 in order to prevent overfitting and deal with weird loss graphs by decreasing the learning rate, and a warmup for the learning rate schedule was also included.

A standard training loop was used for around 30 epochs that plotted the loss values and counted the number of errors for each sentence type.

![Plotted loss values from training loop. They seem to be somewhat erratic.](https://cdn-images-1.medium.com/max/2000/0*eMW5qtRcfnxHe32Y.png)*Plotted loss values from training loop. They seem to be somewhat erratic.*

Audio augmentation was also applied in the training loop for the trials with audio augmentation, using audiomentations with strictly PitchShift (min_semitones = -2, max_semitones= 2, p = 0.1). TimeShift seems to cause false positives and false negatives due to the nature of the speech impediments.

At the end of the training loop, the scheduler increases by one step and Sci-kit Learn library is used to print a classification report detailing the recall, precision, f1-score, and support for the two labels.

![Example classifiaction-report from sci-kit learn while training the model.](https://cdn-images-1.medium.com/max/2000/0*RvyISsc-wN6bvYYU.png)*Example classifiaction-report from sci-kit learn while training the model.*

### **Validation**

A standard evaluation function was used that takes 4 arguments, the model, dataloader, criterion, and device. It then sets the model to evaluation mode with model.eval(). It iterates over the loss dataset and adds to the running loss while adding incorrectly classified files to the wrong_files list.

### **Error Analysis**

Incorrectly classified files from the validation set were used to determine which sentences were most problematic for classifying SP. Overall, the sentences that were most problematic for classification were the phrases, or the sounds.

![Chart of errors broken down by sentence for wav2vec + audio aug.](https://cdn-images-1.medium.com/max/2000/1*FAI8o6F3-iM6-L32QPsSTQ.png)*Chart of errors broken down by sentence for wav2vec + audio aug.*

### **Results**

![Wav2vec without audio augmentations, highest validation accuracy](https://cdn-images-1.medium.com/max/2000/1*OKNZs0h300fkloPyQJMBTg.png)*Wav2vec without audio augmentations, highest validation accuracy*

![Wav2vec with audio augmentations — although validation accuracy is lower, model is likely more generalizable](https://cdn-images-1.medium.com/max/2000/1*lqWbLgvKp8IF6n_RlDa5Sw.png)*Wav2vec with audio augmentations — although validation accuracy is lower, model is likely more generalizable*

![OpenAI Whisper (no audio augmentation)—slightly lower validation accuracy, but still respectable. Sidenote: there’s a bug with the misclassified files list, but the total number of missclassified files is correct.](https://cdn-images-1.medium.com/max/2460/1*EVJM-7yEB1dIaOlW8xnCdA.png)*OpenAI Whisper (no audio augmentation)—slightly lower validation accuracy, but still respectable. Sidenote: there’s a bug with the misclassified files list, but the total number of missclassified files is correct.*

## Deployment

The model was deployed on Huggingface Inference API using Gradio. This makes the model accessible for anybody wanting to test if they have Pseudobulbar palsy or other neurodegenerative diseases that also cause dysarthria. The deployment can be viewed here: [https://huggingface.co/spaces/spycoder/wav2vec](https://huggingface.co/spaces/spycoder/wav2vec)

The gradio app.py file can be viewed here: [https://huggingface.co/spaces/spycoder/wav2vec/blob/main/app.py](https://huggingface.co/spaces/spycoder/wav2vec/blob/main/app.py)

It includes an option for uploading a .wav file for prediction, as well as an option for recording directly from the microphone. Users should upload recordings of one of the 8 sentences that the model was trained on, which are also listed on the interface.

![Screenshot of the deployment on Huggingface!](https://cdn-images-1.medium.com/max/3098/1*HQLPgjEAnD26-D7C4mvW9Q.png)*Screenshot of the deployment on Huggingface!*

## Overall Comments on AI Builders

I send my gratitude to Ajarn Mild and P’ Name for guiding me along this process for the past few months. I learned a lot from doing this project, from the big picture of artificial intelligence to the nitty-gritty of cross-entropy loss functions and quantization. I really enjoyed the family-like environment and the occasional discord games we played long after everyone else had left their calls. I hope that I’ll be able to use this newfound knowledge as a foundation for my future projects in AI/ML, and I’m already excited to learn what I haven’t yet discovered.

On the topic of this project specifically, I think that it could have a large impact if the dysarthria from PP caused by stroke is generalizable to dysarthria from neurological diseases. Although the cause of the dysarthria in this dataset is not a common neurological diseases, this project acts as a sound proof of concept for Thai language audio classification of various neurodegenerative diseases that cause dysarthria. Compared to baselines, the validation accuracy of the Wav2Vec and Whisper models are relatively high after audio augmentation + learning rate scheduling with warmup.

Next steps would be to do cross-testing with diseases like Parkinsons and Alzheimers to see if the model still works or to collect speech data from people with those neurodegenerative diseases if the dysarthria is not generalizable. One suggestion from P’Name would be to try training on models already fine-tuned to the Thai language. The potential impact of an accessible early diagnosis of these neurodegenerative diseases is large and could benefit many people by alerting them of their conditions and prompting them to find medical care for early intervention.
