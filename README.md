# TensorFlow graph panel for Grafana to create and train Machine Learning models

![TensorFlow Panel in Grafana](docs/tensorflow_panel.png)

Grafana graph panel with TensorFlow models has a capability to create a train
ML models right in web-browser. No backend required. Panel uses TensoFlow JS - an open-source hardware-accelerated JavaScript library for training and deploying machine learning models.

# Installation

 * Plugin should be placed in `.../grafana/data/plugins`
 * git clone https://github.com/vsergeyev/tensorflow-grafana-app.git
 * cd tensorflow-grafana-app
 * yarn
 * yarn dev --watch
 * restart Grafana
 * LoudML app should be in plugins list, you may need to activate it
 * enjoy :)
 
# Workflow

 * Select data from your prefered datasource
 * Select TensorFlow graph as Visualization
 * Select data range to train/fit model on
 * Click "Create TensorFlow model" button
 * Enjoy :)
