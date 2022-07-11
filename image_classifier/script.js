const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];


ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}


function enableCam() {
    if(hasGetUserMedia()) {
        const constraints = {
            video:true,
            width: 640,
            height: 480,
        }
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
            VIDEO.srcObject = stream
            VIDEO.addEventListener('loadeddata',function() {
                videoPlaying = true
                ENABLE_CAM_BUTTON.classList.add('removed')
            })
        })
    } else {
        console.warn('getUserMedia() is not supported by your browser')
    }
}

/**
 * Purge data and start over. Note this does not dispose of the loaded
 * MobileNet model and MLP head tensors as you will need to reuse
 * them to train a new model.
 **/
function reset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
        trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;
    STATUS.innerText = 'No data collected';

    console.log('Tensors in memory: ' + tf.memory().numTensors);
}


function predictLoop() {
    if (predict) {
        tf.tidy(function() {
            let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255)
            let resied_tensor_frame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],true)
            let image_features = mobilenet.predict(resied_tensor_frame.expandDims())
            let prediction = model.predict(image_features).squeeze()
            let highestIndex = prediction.argMax().arraySync()
            let prediction_array = prediction.arraySync()

            STATUS.innerText = `Prediction: ${CLASS_NAMES[highestIndex]} with ${Math.floor(prediction_array[highestIndex]*100)}% confidence`
        })

        window.requestAnimationFrame(predictLoop)
    }
}

async function trainAndPredict() {
    predict = false
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs)
    let outputs_as_tensor = tf.tensor1d(trainingDataOutputs, 'int32')
    let one_hot_outputs = tf.oneHot(outputs_as_tensor, CLASS_NAMES.length)
    let inputs_as_tensor = tf.stack(trainingDataInputs)

    let results = await model.fit(inputs_as_tensor, one_hot_outputs, {
            shuffle: true,
            batchSize: 5,
            epochs: 10,
            callbacks: {onEpochEnd: logProgress}
        })
    outputs_as_tensor.dispose()
    one_hot_outputs.dispose()
    inputs_as_tensor.dispose()
    predict = true
    predictLoop()
}

function logProgress(epoc, logs) {
    console.log('Data for epoch ' + epoc, logs)
}



let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
    dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
    dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
    // Populate the human readable names for classes.
    CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}


function dataGatherLoop() {
    if(videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
        let imageFeatures = tf.tidy(() =>{
            let videoFrameAsTensor = tf.browser.fromPixels(VIDEO)
            let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],true)
            let normalizedTensorFrame = resizedTensorFrame.div(255)
            return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze()
        })
        trainingDataInputs.push(imageFeatures)
        trainingDataOutputs.push(gatherDataState)

        if(examplesCount[gatherDataState] === undefined) {
            examplesCount[gatherDataState] = 0
        }
        examplesCount[gatherDataState]++

        STATUS.innerText = ''
        for(let n=0; n<CLASS_NAMES.length; n++) {
            STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '.'
        }
        window.requestAnimationFrame(dataGatherLoop)
    }
}

function gatherDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'))
    gatherDataState = (gatherDataState === STOP_DATA_GATHER)?classNumber:STOP_DATA_GATHER
    dataGatherLoop()
}


let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;



async function loadMobileNetFeatureModel() {
    const URL =
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

    mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
    STATUS.innerText = 'MobileNet v3 loaded successfully!';

    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
        let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        console.log('shape',answer.shape);
    });
}

loadMobileNetFeatureModel();



let model = tf.sequential()
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}))
model.add(tf.layers.dense({units:CLASS_NAMES.length, activation: 'softmax'}))
model.summary()

model.compile({
    optimizer:'adam',
    loss: (CLASS_NAMES===2)?'binaryCrossentropy':'categoricalCrossentropy',
    metrics:['accuracy']
})
