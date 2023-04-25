import React from "react";
import { useState, useEffect } from "react";
import { MdCloudUpload, MdDelete } from "react-icons/md";
import { AiFillFileImage } from "react-icons/ai";
import axios from "axios";
import Visualizer from "../Visualizer/index";
import { Buffer } from 'buffer';
import "./index.css";

const synthesiaData = {
  category: [
    "aortic",
    "arterial",
    "abnorm",
    "lung",
    "spinal",
    "cardiac",
    "inter",
  ],
  answer: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
};

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState("No selected file");
  const [response, setResponse] = useState(null);
  const [encodedImage, setEncodedImage] = useState(null);
  return (
    <main className="input-main grid grid-cols-2">
      <div className="col-span-1 h-80">
        <form
          className="overflow-auto w-full h-full"
          action=""
          onClick={() => document.querySelector(".input-field").click()}
        >
          <input
            type="file"
            accept="image/*"
            className="input-field"
            hidden
            onChange={({ target: { files } }) => {
              files[0] && setFileName(files[0].name);
              const image = files[0];
              if (image) {
                setImage(URL.createObjectURL(image));
                const formData = new FormData();
                formData.append("image", image);
                axios({
                  method: "post",
                  url: "/api/v1/CXRNet/predict",
                  headers: {
                    "Content-Type": "multipart/form-data",
                  },
                  data: formData,
                })
                  .then((response) => {
                    setEncodedImage(`data:image/jpeg+jpg+png;base64,${response.data.gradcam}`);
                    setResponse(response.data);
                  })
                  .catch((error) => {
                    console.log(error);
                  });
              }
            }}
          />
          {encodedImage ? (
            <img src={encodedImage} width={500} height={300} alt={fileName} />
          ) : (
            <>
              <MdCloudUpload color="#1475cf" size={60} />
              <p>Browse Files to Upload</p>
            </>
          )}
        </form>

        <section className="uploaded-row">
          <AiFillFileImage color="#1457cf" />
          <span className="upload-content">
            {fileName}
            <MdDelete
              onClick={() => {
                setFileName("No selected File");
                setImage(null);
              }}
              />
          </span>
        </section>
      </div>
      <div className="">
        {response ? (
          <Visualizer data={response} />
          ) : (
            <Visualizer data={synthesiaData} />
            )}
      </div>
    </main>
  );
}

export default ImageUploader;
