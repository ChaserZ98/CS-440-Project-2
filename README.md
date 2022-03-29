<div id="top"></div>

[![commits](https://badgen.net/github/commits/ChaserZ98/Face-and-Digit-Classification/main)](https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/graphs/commit-activity)
[![forks](https://badgen.net/github/forks/ChaserZ98/Face-and-Digit-Classification)](https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/network/members)
[![stars](https://badgen.net/github/stars/ChaserZ98/Face-and-Digit-Classification)](https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/stargazers)
[![issues](https://badgen.net/github/issues/ChaserZ98/Face-and-Digit-Classification)](https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/issues/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

<div align="center">
  <h1 align="center">Face and Digit Classification</h1>
  <p align="center">
    A implementation of Naive Bayes algorithm and perceptron algorithm
    <br />
    <a href="https://GitHub.com/ChaserZ98/Face-and-Digit-Classification"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://GitHub.com/ChaserZ98/Face-and-Digit-Classification">View Demo</a>
    ·
    <a href="https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/issues">Report Bug</a>
    ·
    <a href="https://GitHub.com/ChaserZ98/Face-and-Digit-Classification/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built with</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#test-result">Test result</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

This is a course project of Rutgers CS440 Intro to Artificial Intelligence.

### Built with

* Language:
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Dataset
The `data` folder contains two different types of images: digits from 0-9 and human faces.

Each type of data is splited to three parts:
* Training dataset
* Validation dataset
* Test dataset

Size of a digit image is 28 x 28.

Size of a face image is 60 x 70.

### Test result

#### Average training time
* Digit

  <img width=550 height=350 alt="Digit Average Training Time" title="Digit Average Training Time" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Digit%20Average%20Training%20Time.jpg?raw=true"/>
* Face

  <img width=550 height=350 alt="Face Average Training Time" title="Face Average Training Time" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Face%20Average%20Training%20Time.jpg?raw=true"/>

#### Accuracy and standard deviation
Digit using Naive Bayes

<img width=550 height=350 alt="Accuracy Digit(Naive Bayes)" title="Accuracy Digit(Naive Bayes)" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Accuracy%20Digit%20(Naive%20Bayes).jpg?raw=true"/>

Digit using Perceptron

<img width=550 height=350 alt="Accuracy Digit(Perceptron)" title="Accuracy Digit(Perceptron)" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Accuracy%20Digit%20(Perceptron).jpg?raw=true"/>

Face using Naive Bayes

<img width=550 height=350 alt="Accuracy Face(Naive Bayes)" title="Accuracy Face(Naive Bayes)" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Accuracy%20Face%20(Naive%20Bayes).jpg?raw=true"/>

Face using Perceptron

<img width=550 height=350 alt="Accuracy Face(Perceptron)" title="Accuracy Face(Perceptron)" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Accuracy%20Face%20(Perceptron).jpg?raw=true"/>

#### Weight visualization examples
* Digit
  
  <img width=350 height=350 alt="Weight for digit 1" title="Weight for digit 1" src = "https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Weight%20Visualization%20Digit%201.jpg?raw=true"/>

  <img width=350 height=350 alt="Weight for Digit 3" title="Weight for Digit 3" src = "https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Weight%20Visualization%20Digit%203.jpg?raw=true"/>
* Face
  
  <img width=350 height=350 alt="Weight for face" title="Weight for face" src="https://github.com/ChaserZ98/Face-and-Digit-Classification/blob/main/LaTeX/Weight%20Visualization%20Face.jpg?raw=true"/>

## Usage

Classify face and digit by using naive bayes algorithm and perceptron algorithm.

## License

Distributed under the MIT License.

See `LICENSE` for more information.

## Contact

Feiyu Zheng - [feiyuzheng98@gmail.com](mailto:feiyuzheng98@gmail.com)

Project Link: [https://github.com/ChaserZ98/Face-and-Digit-Classification](https://github.com/ChaserZ98/Face-and-Digit-Classification)

<p align="right">(<a href="#top">back to top</a>)</p>