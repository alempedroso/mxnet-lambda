# MXNet package for AWS Lambda

This is a reference application that predict labels for an image, using a pre-built model on MXNet deployed on [AWS Lambda](https://aws.amazon.com/lambda). You can leverage the precompiled libraries to build your prediction pipeline on Lambda.

## Components

- MXNet
- OpenCV
- Numpy

## Instructions

- Create a Lambda function from the CLI by running the following commands: 

```
cd mxnet-lambda/src
zip -9r lambda_function.zip  * 
aws lambda update-function-code --function-name opencv-lambda --zip-file fileb://lambda_function.zip

```

## Notes

All the necessary libraries needed for MXNet have been copied to the src/lib folder. In addition, OpenCV for Python is also available for your use.  
