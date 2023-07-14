//import { readFile } from 'fs/promises';
const express = require("express");
const bodyParser = require('body-parser');
const cors = require('cors')
const axios = require('axios')
const {Buffer} = require('node:buffer')
const  {readFileSync} = require('fs')
//const { decodeImage } = require('@tensorflow/tfjs-node');

const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");


// async function loadModel(){
//     const handler = tfn.io.fileSystem('model.json');
//     const model = await tf.loadLayersModel(handler);
//     console.log("Model loaded")
// }


const PORT = process.env.PORT || 3003;
const app = express();

app.use(cors());
//app.use(bodyParser.json());
// Configurar el límite de tamaño de carga
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
//app.use(express.json());

const json = JSON.parse(readFileSync('model.json'))

// const json = JSON.parse(
//   await readFile(
//     new URL('./model.json', import.meta.url)
//   )
// );

app.get("/", async (req, res) => {
    //res.send("Api de personajes, para usar use un número en la url ejemplo: url/12")

    //res.send(json);
    // const url = 'https://cdn.shopify.com/s/files/1/0063/7324/5012/products/la-libreteria-sacapuntas-kum-100-70-03_700x.jpg?v=1634797691'

    // const response = await axios.get(url, { 
    //     responseType: 'arraybuffer' 
    //     })
    // const imageData = new Uint8Array(response.data);
    try{

      const model = await tf.loadLayersModel('file://./model.json');
      console.log(model)
      res.json({name: 'api para el modelo'})

    } catch (err) {
      console.error(err);
      res.status(500).send('Error');
    }
})

app.post('/predict', async (req, res) => {
    try {
        // Carga el modelo
        const model = await tf.loadLayersModel('file://./model.json');
    
        // Convierte los datos de entrada a tensor y redimensiona
        const input = tf.tensor(req.body).reshape([1, 224, 224, 3]);
    
        // Realiza la predicción
        const output = model.predict(input);
        const prediction = output.dataSync();
    
        // Envía la predicción como respuesta
        res.send({ prediction });
      } catch (err) {
        console.error(err);
        res.status(500).send('Error');
      }
});

app.listen(PORT, () => {
    console.log("servicio corriendo en puerto: ", PORT)
} )