import './App.css';
import ImageUploader from './component/uploader';


function App() {
  return (
    <div className="App">
      <h1>Storage Age Identification of Dried Tangerine Peels with Machine Learning and Imaging</h1>
      <p>Citri Reticulatae Pericarpium (Chen Pi in Chinese) are sun-dried mandarin tangerine peels that are largely used in traditional Chinese medicine. Their medicinal value, and accordingly their market price, is correlated with their age. However, commercially, their storage age is not apparent to the average consumer. Our project aims to identify the storage age of dried tangerine peels through smartphone-based imaging and machine learning. ​</p>
     <br></br>
     <p>Upload an image of a dried tangerine peel below to predict its storage age.</p>
      <ImageUploader/>

    </div>
  );
}

export default App;
