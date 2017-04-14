
This is a collection of CNN examples from theano and deeplearning4j frameworks.


NOTICE1: the MINST dataset is used for the theanl CNN.

NOTICE2: the dl4j example is for animal classifications.

NOTICE3: Do NOT change pom.xml files in the deeplearning4j.

NOTICE4: The Makefile is based on https://docs.docker.com/opensource/project/set-up-dev-env/


Please follow the instructions below to run the CNN examples.


[1] download (or git clone) this source code folder

[2] cd downloaded-source-code-folder

[3] sudo make BIND_DIR=. shell

	wait ... wait ... then a bash shell will be ready (root@c909bdd3f1aa:/#)



[4 theano instructions] 

[4-1] root@ec69b4a68f49:/# cd /root/cnn

[4-2] root@ec69b4a68f49:~/cnn# cd python/

[4-3] root@ec69b4a68f49:~/cnn/python# cd Theano/

[4-4] root@ec69b4a68f49:~/cnn/python/Theano# python setup.py develop

[4-5] root@ec69b4a68f49:~/cnn/python/Theano# cd ..

[4-6] root@ec69b4a68f49:~/cnn/python# cd ..

[4-7] root@ec69b4a68f49:~/cnn# cd cnn_examples/

[4-8] root@ec69b4a68f49:~/cnn/cnn_examples# cd theano/

[4-9] root@ec69b4a68f49:~/cnn/cnn_examples/theano# python ./mnist_cnn_01.py

[4-10] the output may look like

	
	testing_size = 10000
	testing_size = 10000
	... labels ... [0] 7
	... labels ... [1] 2
	... labels ... [2] 1
	... labels ... [3] 0
	... labels ... [4] 4
	... labels ... [5] 1
	... labels ... [6] 4
	... labels ... [7] 9
	... labels ... [8] 5
	... labels ... [9] 9
	... labels ... [10] 0
	... labels ... [11] 6
	... labels ... [12] 9
	... labels ... [13] 0
	... labels ... [14] 1
	... labels ... [15] 5
	... labels ... [16] 9
	... labels ... [17] 7
	... labels ... [18] 3
	... labels ... [19] 4
	len(training_img) =  47040000
	len(training_img)/(rows*cols) = 60000
	.. training_size = 60000
	.. rows = 28
	.. cols = 28
	.. len(ml_training_img) = 39200000
	.. len(ml_training_img)/(rows*cols) = 50000.000000
	.. len(ml_valid_img) = 7840000
	.. len(ml_valid_img)/(rows*cols) = 10000.000000
	.. len(traing_lbl) = 60000

	.. training_size = 60000

	type(test_set_y)=  <class 'theano.tensor.var.TensorVariable'>
	test_set_y.type()=  <TensorType(int32, vector)>
	
	
	   batch_size = 500 
	   n_train_batches = 100 
	   n_valid_batches = 20 
	   n_test_batches  = 20 
	
	... building the model, so wait for a while
	./mnist_cnn_01.py:430: UserWarning: DEPRECATION: 
	the 'ds' parameter is not going to exist anymore as it is going to be replaced by the parameter 'ws'.
	  ignore_border=True
	
	type(temp_x) =  <class 'theano.tensor.var.TensorVariable'>
	temp_x.type() =  <TensorType(float64, matrix)>
		
	type(temp_y) =  <class 'theano.tensor.var.TensorVariable'>
	temp_y.type() =  <TensorType(int32, vector)>
	
	... training, so wait again
	('training @ iter = ', 0)
	('training @ iter = ', 10)
	('training @ iter = ', 20)
	('training @ iter = ', 30)
	('training @ iter = ', 40)
	('training @ iter = ', 50)
	('training @ iter = ', 60)
	('training @ iter = ', 70)
	('training @ iter = ', 80)
	('training @ iter = ', 90)
	epoch 1, minibatch 100/100, validation error 5.200000 
	     epoch 1, minibatch 100/100, test error of best model 5.480000 
	('training @ iter = ', 100)
	('training @ iter = ', 110)
	('training @ iter = ', 120)
	('training @ iter = ', 130)
	('training @ iter = ', 140)
	('training @ iter = ', 150)
	('training @ iter = ', 160)
	('training @ iter = ', 170)
	('training @ iter = ', 180)
	('training @ iter = ', 190)
	epoch 2, minibatch 100/100, validation error 3.310000 
	     epoch 2, minibatch 100/100, test error of best model 3.450000 
	Optimization complete.
	Best validation score of 3.310000 % obtained at iteration 200, with test performance 3.450000
	


[5 dl4j instructions]


	This source code includes 
	javacpp
	javacpp-presets (for OpenCV and OpenBLAS)
	libnd4j
	nd4j
	datavec
	deeplearning4j
	exmaples (AnimalsClassification java example code)


[5-1] root@8690c5f919ee:/# cd /root/cnn/

[5-2] root@8690c5f919ee:~/cnn# cd java

[5-3] root@8690c5f919ee:~/cnn/java# cd dl4j

[5-4] root@8690c5f919ee:~/cnn/java/dl4j# cd javacpp

[5-5] root@8690c5f919ee:~/cnn/java/dl4j/javacpp# mvn clean install

[5-6] root@8690c5f919ee:~/cnn/java/dl4j/javacpp# cd ..

[5-7] root@8690c5f919ee:~/cnn/java/dl4j# cd javacpp-presets/

[5-8] root@8690c5f919ee:~/cnn/java/dl4j/javacpp-presets# ./cppbuild.sh

[5-9] root@8690c5f919ee:~/cnn/java/dl4j/javacpp-presets# mvn clean install

[5-10] root@8690c5f919ee:~/cnn/java/dl4j/javacpp-presets# cd ..

[5-11] root@8690c5f919ee:~/cnn/java/dl4j# cd libnd4j/

[5-12] root@8690c5f919ee:~/cnn/java/dl4j/libnd4j# ./buildnativeoperations.sh

[5-13] root@7778850ee17b:~/cnn/java/dl4j/libnd4j# export LIBND4J_HOME=`pwd`

[5-14] root@8690c5f919ee:~/cnn/java/dl4j/libnd4j# cd ..

[5-15] root@8690c5f919ee:~/cnn/java/dl4j# cd nd4j/

[5-16] root@8690c5f919ee:~/cnn/java/dl4j/nd4j# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'

[5-17] root@8690c5f919ee:~/cnn/java/dl4j/nd4j# cd ..

[5-18] root@8690c5f919ee:~/cnn/java/dl4j# cp -rp ./nd4j/ /usr/local/lib/

[5-19] root@8690c5f919ee:~/cnn/java/dl4j# cd datavec/

[5-20] root@8690c5f919ee:~/cnn/java/dl4j/datavec# bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true

[5-21] root@8690c5f919ee:~/cnn/java/dl4j/datavec# cd ..

[5-22] root@8690c5f919ee:~/cnn/java/dl4j# cd deeplearning4j/

[5-23] root@7778850ee17b:~/cnn/java/dl4j/deeplearning4j# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0'

[5-24] root@8690c5f919ee:~/cnn/java/dl4j/deeplearning4j# cd ..

[5-25] root@8690c5f919ee:~/cnn/java/dl4j# cd ..

[5-26] root@8690c5f919ee:~/cnn/java# cd ..

[5-27] root@8690c5f919ee:~/cnn# cd cnn_examples/

[5-28] root@8690c5f919ee:~/cnn/cnn_examples# cd deeplearning4j/

[5-29] root@8690c5f919ee:~/cnn/cnn_examples/deeplearning4j# cd dl4jCNN/

[5-30] root@8690c5f919ee:~/cnn/cnn_examples/deeplearning4j/dl4jCNN# mvn clean compile

[5-31] root@7778850ee17b:~/cnn/cnn_examples/deeplearning4j/dl4jCNN# mvn clean package 

[5-32] root@8690c5f919ee:~/cnn/cnn_examples/deeplearning4j/dl4jCNN# java -Djava.library.path=/usr/lib/:/usr/lib/gcc/x86_64-linux-gnu/5/:/usr/local/lib/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-x86_64:/root/cnn/java/dl4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/target/classes/org/nd4j/nativeblas/linux-x86_64:/root/cnn/java/dl4j/javacpp-presets/openblas/target/classes/org/bytedeco/javacpp/linux-x86_64/:/root/cnn/java/dl4j/javacpp-presets/opencv/target/classes/org/bytedeco/javacpp/linux-x86_64/ -cp /root/.m2/repository/org/datavec/datavec-data-image/0.7.3-SNAPSHOT/datavec-data-image-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/datavec/datavec-api/0.7.3-SNAPSHOT/datavec-api-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/slf4j/slf4j-api/1.7.12/slf4j-api-1.7.12.jar:/root.m2/repository/org/slf4j/slf4j-simple/1.7.12/slf4j-simple-1.7.12.jar:/root/.m2/repository/org/nd4j/nd4j-jackson/0.7.3-SNAPSHOT/nd4j-jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-common/0.7.3-SNAPSHOT/nd4j-common-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-context/0.7.3-SNAPSHOT/nd4j-context-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-buffer/0.7.3-SNAPSHOT/nd4j-buffer-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-api/0.7.3-SNAPSHOT/nd4j-api-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-core/0.7.3-SNAPSHOT/deeplearning4j-core-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-nn/0.7.3-SNAPSHOT/deeplearning4j-nn-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/deeplearning4j/deeplearning4j-nlp/0.7.3-SNAPSHOT/deeplearning4j-nlp-0.7.3-SNAPSHOT.jar::/root/.m2/repository/org/nd4j/nd4j-native-platform/0.7.3-SNAPSHOT/nd4j-native-platform-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/nd4j-native/0.7.3-SNAPSHOT/nd4j-native-0.7.3-SNAPSHOT.jar:/root/.m2/repository/commons-lang/commons-lang/2.6/commons-lang-2.6.jar::/root/.m2/repository/commons-io/commons-io/1.3.2/commons-io-1.3.2.jar:/root/.m2/repository/org/apache/commons/commons-compress/1.8/commons-compress-1.8.jar::/root/.m2/repository/org/apache/commons/commons-math3/3.6/commons-math3-3.6.jar:/root/.m2/repository/org/nd4j/nd4j-native-api/0.7.3-SNAPSHOT/nd4j-native-api-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/nd4j/jackson/0.7.3-SNAPSHOT/jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/com/google/collections/google-collections/1.0/google-collections-1.0.jar:/root/.m2/repository/org/reflections/reflections/0.9.10/reflections-0.9.10.jar:/root/.m2/repository/org/bytedeco/javacpp/1.3/javacpp-1.3.jar::/root/.m2/repository/commons-codec/commons-codec/1.10/commons-codec-1.10.jar:/root/.m2/repository/org/bytedeco/javacpp-presets/openblas/0.2.19-1.3/openblas-0.2.19-1.3.jar:/root/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.1.0-1.3/opencv-3.1.0-1.3.jar:/root/.m2/repository/org/bytedeco/javacpp-presets/1.3/javacpp-presets-1.3.jar:/root/deeplearning/nd4j/nd4j-shade/jackson/target/jackson-0.7.3-SNAPSHOT.jar:/root/.m2/repository/org/apache/commons/commons-lang3/3.5/commons-lang3-3.5.jar:/root/.m2/repository/commons-io/commons-io/2.5/commons-io-2.5.jar:/root/.m2/repository/com/google/guava/guava/21.0/guava-21.0.jar:/root/.m2/repository/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar:/root/.m2/repository/javassist/javassist/3.12.1.GA/javassist-3.12.1.GA.jar:/root/.m2/repository/org/bytedeco/javacv/1.3.1/javacv-1.3.1.jar:/root/.m2/repository/org/datavec/datavec-nd4j-common/0.7.3-SNAPSHOT/datavec-nd4j-common-0.7.3-SNAPSHOT.jar:/root/cnn/cnn_examples/deeplearning4j/dl4jCNN/target/dl4jCNN-1.0-SNAPSHOT.jar com.mycompany.project.App


[5-33] the output may look like 

	
	SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
	SLF4J: Defaulting to no-operation (NOP) logger implementation
	SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
	... Load data ...
	... Build model ...
	... Train model ...
	
	Training on transformation: class org.datavec.image.transform.FlipImageTransform
	Training on transformation: class org.datavec.image.transform.WarpImageTransform
	Training on transformation: class org.datavec.image.transform.FlipImageTransform
        
	For a single example that is labeled bear the model predicted turtle
	... Example finished ...

	


