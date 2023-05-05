import React, { useState } from 'react';
import { Loading } from './loader.js'

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    setLoading(true);
    e.preventDefault();
    const formData = new FormData();
    formData.append('image', image);
    // replace the URL below with the API endpoint for uploading images
    fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        switch (data.prediction) {
          case 0:
            alert(`Response: 6 years`);
            break;
          case 1:
            alert(`Response: 10 years`);
            break;
          case 2:
            alert(`Response: 15 years`);
            break;
          case 3:
            alert(`Response: 15 years`);
            break;
        }
      })
      .catch((error) => {
        setLoading(false);
        alert(`Error: ${error.message}`);
      });
  };

  const handleChange = (e) => {
    setImage(e.target.files[0]);
  };
  if (loading) {
    return <Loading/>
  }
  return (
    <form onSubmit={handleSubmit}>
      <input type="file" onChange={handleChange} className="image_input"/>
      <br></br>
      {image && <img src={URL.createObjectURL(image)} alt="uploaded image" />}
      <button type="submit" disabled={!image}>
        Upload
      </button>
    </form>
  );
}

export default ImageUploader;