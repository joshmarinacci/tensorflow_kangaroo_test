/*

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4



In this example we load data correlating horsepower and miles per gallon.
Then we will train a neural net from the data to be able to predict
MPG from new horsepower

 */

async function getData() {
    const carsDataResponse = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json")
    const carsData = await carsDataResponse.json()
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));
    return cleaned
}

async function run() {
    console.log("getting data")
    const data = await getData()
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    // plot the training data. no ML here
    tfvis.render.scatterplot(
        {name:'Horsepower v MPG'},
        {values},
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height: 300,
        }
    );
    console.log(data)

    const model = createModel()
    // show info about the model. No training done yet.
    tfvis.show.modelSummary({name: 'Model Summary'},model)

    // convert to tensors
    const tensorData = convertToTensor(data)
    const {inputs, labels} = tensorData
    await trainModel(model, inputs, labels)
    console.log("done training")

}

function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data)
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg)
        //convert to 2d tensor
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensor = tf.tensor2d(labels, [labels.length, 1])

        // normalize to the range of 0 - 1 using min-max scaling
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }

    })
}

async function trainModel(model, inputs, labels) {
    //compile the model first using the ADAM optmizer
    // and mean squared error as the loss function
    // metrics will be mean squared error (mse)
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics:['mse'],
    })
    // 32 items per iteration
    // 50 iterations
    const batch_size = 32
    const epocs = 50

    // start the training
    // use a callback so we can see the progress
    return await model.fit(inputs, labels, {
        batchSize: batch_size,
        epochs: epocs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Perf'},
            ['loss','mse'],
            {height: 200, callbacks: ['onEpochEnd']}
        )
    })
}


function createModel() {
    // make a new sequential model. It's sequential because inptus flow straight
    // down to the output. Other kinds of models can have branches and loops.
    // we start with the simplest possible one
    const model = tf.sequential()
    // add an input layer connected to a dense layer with one hidden unit (units:1)
    // a 'dense' layer multiplies it's inputs by a matrix (weights) then adds a number (bias)
    // inputShape:[1] is because our input has a single number per item (the HP of a car)
    // units: sets how big the weight matrix will be for this layer. 1 means 1 weight for each input
    model.add(tf.layers.dense({inputShape:[1],units:1,useBias:true}))
    // add an output layer. units:1 because we want to output one number
    model.add(tf.layers.dense({units:1, useBias: true}))
    return model
}


document.addEventListener('DOMContentLoaded', () => {
    console.log("loaded")
    run()
})

