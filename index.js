const webcamElement = document.getElementById('webcam');
async function setupWebcam() {
	return new Promise((resolve, reject) => {
		const navigatorAny = navigator;
		navigator.getUserMedia = navigator.getUserMedia ||
			navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
			navigatorAny.msGetUserMedia;
		if (navigator.getUserMedia) {
			navigator.getUserMedia({video: true},
				stream => {
					webcamElement.srcObject = stream;
					webcamElement.addEventListener('loadeddata',  () => resolve(), false);
				},
				error => reject());
		} else {
			reject();
		}
	});
}


let net;
const classifier = knnClassifier.create();
async function app() {
	console.log('[INFO] Loading mobilenet..');

	// Load the model.
	net = await mobilenet.load();
	console.log('[INFO] Sucessfully loaded model');

	await setupWebcam();

	// Reads an image from the webcam and associates it with a specific class
	// index.
	const classes = ['A', 'B', 'C', '0'];
	const addExample = classId => {
		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
		// to the KNN classifier.
		const activation = net.infer(webcamElement, 'conv_preds');
		// Pass the intermediate activation to the classifier.
		classifier.addExample(activation, classId);
		console.log("added classId:" + classId + " class:" + classes[classId]);
	};

	// When clicking a button, add an example for that class.
	document.getElementById('class-a').addEventListener('click', () => addExample(0));
	document.getElementById('class-b').addEventListener('click', () => addExample(1));
	document.getElementById('class-c').addEventListener('click', () => addExample(2));
	document.getElementById('class-0').addEventListener('click', () => addExample(3));

	while (true) {
		if (classifier.getNumClasses() > 0) {
			// Get the activation from mobilenet from the webcam.
			const activation = net.infer(webcamElement, 'conv_preds');
			// Get the most likely class and confidences from the classifier module.
			const result = await classifier.predictClass(activation);

			// var confidences = result.confidences.map(function(each_element){
			// 	return Number(each_element.toFixed(2));
			// });
			for (var i in result.confidences)
			{
				result.confidences[i] = result.confidences[i].toFixed(2);
			}
			var s_confidences = JSON.stringify(result.confidences);
			var s_result = JSON.stringify(result);

			document.getElementById('console').innerText =
				`prediction: ${classes[result.label]}\n`+
				`result.label: ${result.label}\n`+
				`result.classIndex: ${result.classIndex}\n`+
				`result.confidences: ${s_confidences}`;
		}

		await tf.nextFrame();
	}
}
app();
