import { useState } from "react";
import "./styles.css";

const GenImage = () => {
    const [prompt, setPrompt] = useState("");
    const [seed, setSeed] = useState(42);
    const [guidanceScale, setGuidanceScale] = useState(7.5);
    const [numInfSteps, setNumInfSteps] = useState(10);
    const [img, setImg] = useState(null);
    const [promptImg, setPromptImg] = useState(null);

    const cleanFormData = () => {
        setPrompt("");
        setSeed(42);
        setGuidanceScale(7.5);
        setNumInfSteps(5);
    }

    const handleGenerateImage = async (e) => {

        const requestOptions = {
            method: "GET", 
            headers: {"Content-Type": "application/json"}, 
            
        };

        const response = await fetch(`http://localhost:8000/predict/?prompt=${prompt}&num_inference_steps=${numInfSteps}&guidance_scale=${guidanceScale}`, requestOptions);
        
        if (!response.ok){
            
        } else {
            const imageBlob = await response.blob();
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setImg(imageObjectURL);
            setPromptImg(prompt);
            cleanFormData();
        }
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        setImg(null);
        setPromptImg(null);
        handleGenerateImage();
    }

    return (
        <div className="gen-image-container">
            <div className="gen-image-form">
                <h1 className="gen-image-title">Generate Image with Stable Diffuser</h1>
                <form onSubmit={handleSubmit}>
                    <div className="field">
                        <label className="label">Prompt</label>
                        <input
                            type="text"
                            placeholder="Enter your prompt to generate the image"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="input"
                            required
                        />
                    </div>
                    <div className="field">
                        <label className="label">Seed</label>
                        <input
                            type="number"
                            placeholder="Seed"
                            value={seed}
                            onChange={(e) => setSeed(e.target.value)}
                            className="input"
                        />
                    </div>
                    <div className="field">
                        <label className="label">Guidance Scale</label>
                        <input
                            type="number"
                            placeholder="Guidance Scale"
                            value={guidanceScale}
                            onChange={(e) => setGuidanceScale(e.target.value)}
                            className="input"
                        />
                    </div>
                    <div className="field">
                        <label className="label">Number of Inference Steps</label>
                        <input
                            type="number"
                            placeholder="Number of Inference Steps"
                            value={numInfSteps}
                            onChange={(e) => setNumInfSteps(e.target.value)}
                            className="input"
                        />
                    </div>
                    <button className="button is-primary" type="submit">
                        Generate Image
                    </button>
                </form>
            </div>
            <div className="gen-image-preview">
                {img && (
                    <figure>
                        <img src={img} alt="Generated Image" />
                        <figcaption className="gen-image-title">{promptImg}</figcaption>
                    </figure>
                )}
            </div>
        </div>
    );


}

export default GenImage