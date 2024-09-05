
# Agroboost🌱🌱

Agroboost is an innovative project developed during an internship to showcase OCP Group’s commitment to leveraging cutting-edge technology to revolutionize agriculture. By integrating AI and advanced technologies, Agroboost aims to address global agricultural challenges and improve farming practices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Overview](#technical-overview)
3. [Technology Stack](#technology-stack)
4. [Demonstration](#demonstration)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)
10. [Acknowledgements](#acknowledgements)

## Project Overview

Agroboost offers three innovative services designed to enhance agricultural practices through advanced technology and AI. These services include Fertify, Fertibot, and PlantGuard.

🌱 Fertify : 

Fertify is a cutting-edge fertilizer recommendation tool that helps farmers optimize their use of fertilizers based on various factors. those factors ranges from
soil type, crop type, environmental conditions, to recommend the most suitable fertilizers.


🤖 Fertibot :

Fertibot is an intelligent chatbot designed to provide information and support related to OCP Group’s fertilizers. It acts as a virtual assistant to answer queries and offer guidance on fertilizer-related topics. 

🪴 PlantGuard :

PlantGuard is an advanced system for monitoring plant health. It leverages AI and image analysis to detect and address plant disease before they become critical. 

🤝 Assistance Chatbot : 
The Assistance Chatbot provides essential information for easy navigation of Agroboost's services.


## Technical Overview

Agroboost leverages state-of-the-art machine learning (ML) techniques and advanced architectures to deliver its services. Below is a technical description of the methods and technologies used in each service: Fertify, Fertibot, PlantGuard and the assistane chatbot.

### Fertify 

🌱 **Fertify** uses machine learning models to provide optimized fertilizer recommendations. The technical components include:

- **Data Collection and Preprocessing:** 
  - **Data Collection:**
  the dataset was available in kaggle https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction. It contains data of various fertilizers information.This data was by researching various websites and sources. 
  - **Feature Engineering:**
   Extracts relevant features and preprocesses data for model training, including normalization and scaling.

- **Machine Learning Models:**
  - **Classification Models:** Random Forest Classifier was used for Recommending the best fertilizer based on different canditions like sol type and crop type...

- **Model Training and Evaluation:**
  - **Performance Metrics:**  the evaluation of model performance were determined by  accuracy, precision, and recall.

check this notebook for more details: [Fertify Notebook](./EDA/Fertify.ipynb)

### FertiBot

🤖 **Fertibot** is powered by natural language processing (NLP) techniques to interact with users regarding fertilizer-related queries. Key technologies include:

- **NLP Techniques:**
  - **Language Models:** the mistralai/Mixtral-8x7B-Instruct-v0.1 was used for generating human like text based on context provided.
  - **RAG:** retrieval augmented generation was used in order to provide context from pdf documents about OCP's Fertilizer to the language model in order to generate accurate and human like text.
  - **Chunking:**  the text extracted from the pdf documents was splitted, in order to obtain context focused chunks and improve relevance when retrieving.
  - **Vector Database:** Chroma Db was used in order to store embeddings of chunks from pdf documents about OCP's Fertilizer.

  The pdf used are available in OCP Offcial Website :

  [OCP Official Website](https://www.ocpgroup.ma/standard-fertilizers)

  [Fertilizer Pdf](./data/Fertilizers)

### PlantGuard

🪴 **PlantGuard** employs image analysis and AI techniques to monitor plant health and detect disease based on plant's leaf. The technical aspects include:
- **Data Collection and Preprocessing:** 
  - **Data Collection:**
  the dataset was available in kaggle https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  . This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.
- **Image Processing and Analysis:**
  - **Computer Vision Models:** Uses convolutional neural networks (CNNs) for image classification and plant's disease detection.

- **Anomaly Detection:**
  - **Classification Algorithms:** Implements classification algorithms to detect diseases.

### Assistance Chatbot

🤝 **The Assistance Chatbot** provides essential information for easy navigation of Agroboost's services. 
- **Data Collection and Preprocessing:** 
  - **Data Collection:**
  the dataset use for training  the MLP Classifier was manually created, it was orginized into a json file of itents each was carectarized with a tag,  a list of questions, and a list of possible answers.
  [intents.json](./data/intents.json).

- **Machine Learning Models:**
  - **Classification Models:** an mlp classifier was trained on tags prediction task using the available questions. 
  you can check [train.py](./src/train.py) for more informations of the training process.

- **NLP Techniques:**
  - **Tokenization:** the  user's question was tokenized into tokens.
  - **Stemming:** the tokens were stemmed, converted to their root form in order to cover a wide range of words.
  - **Bag_of_Words:** a vector representation of text data for machine learning models. The idea is to create a "bag" of words from a text corpus with a focus on the presence of words.

 ## Technology Stack
 - **Front-End:** Html,Css,Js,Bootstrap, This combination provided a solid foundation for building responsive and interactive web pages.
  - **Integration:** ML model's integration was done using Flask as the web framework. Flask handled the API endpoints, allowing the model to make real-time predictions and process data on the server side.
- **AI Stack:**
  - **Fertify:** sklearn package was used  to develop the fertilizer recommendation system.
  - **FertiBot:** hugging face ecosystem was used for llm calling ,langchain  framework for handling essential task such as  ChromeDb and pdf content handling, beside handling the interaction with the llm.
  - **PlantGuard:** TensorFlow was used as the Framework for developping the CNN Model.
  - **The Assistance Chatbot:** The ChatBot was build using Pytorch Frameworks for the classifier.
## Demonstration
- **Video Demonstration**
To get a comprehensive overview of the website's functionalities and design, please watch our [Video Demonstration](https://www.linkedin.com/feed/update/urn:li:activity:7237570119797788672/).

- **Home Page**
this the link of the [Home Page](https://ihssane5.github.io/Agroboost/).

## Setup and Installation

To get started with Agroboost, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ihssane5/Agroboost
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd Agroboost
   ```

3. **Install Dependencies:**
  Since both pip and conda package manger were used you need to run the following command
   ```bash
   pip install -r requirement.txt
   ```

    ```bash
   conda env create -f agroboost.yml
   ```
   
4. **Activate the virtual environnement:**
   the venv name is 'rag' run the following command to activate it 
   ```bash
   conda activate rag
   ```

5. **Run the Application:**
   replace HUGGINGFACE_API_KEY in the .env file with you huggingface api key make sure you have the read permission

6. **Run the Application:**
   ```bash
   python app.py
   ```
  copy the localhost link provided in your browser

7. **Explore Agroboost:**
  Navigate the different services that agroboost provide,don't forget to let me some comment and feedbacks.


## Contributing

We welcome contributions to Agroboost! If you’d like to contribute in this project or any other one, Contact me, my contact information are listed below.
## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or feedback, please contact:

- **Email:** [ihssanenedjaoui5@gmail.com](mailto:ihssanenedjaoui5@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/ihssanenedjaoui/](https://www.linkedin.com/in/ihssanenedjaoui/)
- **GitHub:** [https://github.com/Ihssane5](https://github.com/Ihssane5)

## Acknowledgements
Special thanks to our supervisor and everyone who provided feedback and support during my internship and the development of this project.

Special thanks to https://templatemo.com for the awesome template🤩
