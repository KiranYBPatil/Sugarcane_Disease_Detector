import React, { useState, useEffect } from "react";

// Use lucide-react (Assumed available in React environments)
const Leaf = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17.5 7.5 19 11c-2.3 2.1-3.6 4.3-4.5 6.4-1.1 2.5-3 4-5.5 4zm0 0l2-2.5"/><path d="M12.5 17.5l-4-3.5"/></svg>
);
const UploadCloud = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 14.5v-3.7A4 4 0 0 1 8 7h6a4 4 0 0 1 4 4v.7a3 3 0 0 1 3 3v2a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3z"/><polyline points="12 17 12 12 10 14 12 12 14 14"/></svg>
);
const Loader2 = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
);

// Map prediction labels to visual classes
const getDiseaseStyle = (label) => {
    switch (label) {
        case 'Healthy':
            return { color: 'text-green-600', bg: 'bg-green-100', border: 'border-green-400' };
        case 'RedRot':
        case 'Mosaic':
            return { color: 'text-red-600', bg: 'bg-red-100', border: 'border-red-400' };
        case 'BacterialBlights':
        case 'Rust':
        case 'Yellow':
            return { color: 'text-yellow-600', bg: 'bg-yellow-100', border: 'border-yellow-400' };
        default:
            return { color: 'text-gray-600', bg: 'bg-gray-100', border: 'border-gray-400' };
    }
};

// Assuming the correct backend URL for deployment
const BACKEND_URL = "http://localhost:8000"; // Use Render URL for live deploy!

export default function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = React.useRef(null);

    const handleFile = (f) => {
        if (!f || !f.type.startsWith('image/')) return;
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const upload = async () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: "POST",
                body: formData,
            });
            
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

            const data = await res.json();
            setResult(data);
        } catch (e) {
            console.error("Error connecting to backend:", e);
            setResult({ prediction: "Connection Error", confidence: 0.0, error: true });
        } finally {
            setLoading(false);
        }
    };

    const style = getDiseaseStyle(result?.prediction);
    const confidencePercent = result ? (result.confidence * 100).toFixed(2) : 0;
    const isError = result?.error;

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4 font-inter">
            <header className="py-6 w-full max-w-lg text-center">
                <h1 className="text-3xl font-extrabold text-green-700 flex items-center justify-center space-x-2">
                    <Leaf className="w-8 h-8"/>
                    <span>Sugarcane Disease Detector</span>
                </h1>
                <p className="text-gray-500 mt-2">ViT and Swin Ensemble Model</p>
            </header>

            <main className="w-full max-w-lg bg-white shadow-xl rounded-xl p-6 md:p-8">
                <div 
                    className={`border-4 border-dashed rounded-lg p-6 text-center transition-colors duration-300 mb-6 ${dragActive ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white hover:border-green-300'}`}
                    onDragEnter={handleDrag} 
                    onDragLeave={handleDrag} 
                    onDragOver={handleDrag} 
                    onDrop={handleDrop}
                >
                    <input 
                        ref={inputRef} 
                        type="file" 
                        accept="image/*" 
                        onChange={(e) => handleFile(e.target.files[0])} 
                        className="hidden" 
                    />
                    
                    {preview ? (
                        <div className="flex flex-col items-center">
                            <img src={preview} alt="Image Preview" className="w-full max-h-80 object-contain rounded-lg shadow-md mb-4"/>
                            <p className="text-sm text-gray-600 truncate max-w-full">{file.name}</p>
                        </div>
                    ) : (
                        <div className="p-8 cursor-pointer" onClick={() => inputRef.current.click()}>
                            <UploadCloud className="w-10 h-10 mx-auto text-green-500 mb-2"/>
                            <p className="text-gray-700 font-medium">Drag & Drop Image Here</p>
                            <p className="text-sm text-gray-500">or click to browse (JPG, PNG)</p>
                        </div>
                    )}
                </div>

                <button 
                    onClick={upload} 
                    disabled={!file || loading}
                    className={`w-full py-3 px-4 rounded-lg font-bold transition-colors duration-200 shadow-md ${
                        !file || loading 
                            ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                            : 'bg-green-600 text-white hover:bg-green-700 hover:shadow-lg'
                    } flex items-center justify-center space-x-2`}
                >
                    {loading && <Loader2 className="w-5 h-5 animate-spin"/>}
                    <span>{loading ? "Analyzing Sugarcane..." : "Detect Disease"}</span>
                </button>

                {result && (
                    <div className={`mt-6 p-4 rounded-lg border-2 ${style.border} ${style.bg} transition-all duration-500 shadow-inner`}>
                        <h3 className={`text-xl font-bold ${isError ? 'text-red-600' : style.color} mb-1 flex items-center justify-between`}>
                            {isError ? 'Detection Failed' : 'Prediction Result'}
                            {!isError && <span className="text-sm text-gray-600">Confidence</span>}
                        </h3>

                        <p className={`text-3xl font-extrabold ${isError ? 'text-red-800' : style.color} mb-4`}>
                            {result.prediction}
                        </p>
                        
                        {!isError && (
                            <>
                                <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div 
                                        className={`h-2.5 rounded-full ${style.color === 'text-green-600' ? 'bg-green-600' : 'bg-red-600'}`} 
                                        style={{ width: `${confidencePercent}%` }}
                                    ></div>
                                </div>
                                <p className={`text-base font-semibold mt-2 ${style.color}`}>
                                    {confidencePercent}%
                                </p>
                            </>
                        )}

                        {isError && (
                            <p className="text-sm text-red-700">Could not connect to the FastAPI backend. Check the server is running on {BACKEND_URL}.</p>
                        )}
                    </div>
                )}
            </main>

            <footer className="text-center text-sm text-gray-400 mt-6">
                &copy; {new Date().getFullYear()} AI Crop Diagnostics. All rights reserved.
            </footer>
        </div>
    );
}
