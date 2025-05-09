import React, { useEffect, useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
const endpoint = 'http://34.171.5.198:8080';
// const endpoint = 'http://localhost:5050';
export default function App() {
  const [ts, setTs] = useState('')
  const [videoReady, setVideoReady] = useState(true)
  const handleTrain = () => {
    const requestBody = {
      modelType: modelType,
      step: step,
      rewardFunction: rewardFunction
    };
    setVideoReady(false);
    fetch(`${endpoint}/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    })
    .then(() => {
      setVideoReady(true);  // Now display the video
    })

    .catch((err) => {
      console.error('Training failed:', err);
    });
}

useEffect(() => {
  if(videoReady === true) {
    fetch(`${endpoint}/video`)
    .then(() => {
      setTs(Date.now())
    })
    .catch((err) => {
      console.error('fetching failed:', err);
    });
  }
}, [videoReady]);

  const [modelType, setModelType] = useState('ppo');
  const [step, setStep] = useState('100,000');
  const [rewardFunction, setRewardFunction] = useState('simple');

  const handleModelChange = (e) => setModelType(e.target.value);
  const handleStepChange = (e) => setStep(e.target.value);
  const handleRewardChange = (e) => setRewardFunction(e.target.value);

  return (
    <div className="container text-center py-4">
      <div className='mb-3'>
        <h1>Play with dAIno!</h1>
      </div>
      <div className="container">
        <div className='row mb-3'>
          <div className='col-4 text-end'>Agent</div>
          <div className='col-5'>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox1"
                value="ppo"
                checked={modelType === 'ppo'}
                onChange={handleModelChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox1">PPO</label>
            </div>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox2"
                value="DQN"
                checked={modelType === 'DQN'}
                onChange={handleModelChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox2">DQN</label>
            </div>
          </div>
        </div>
       
        <div className='row mb-3'>
          <div className='col-4 text-end'>Timesteps</div>
          <div className='col-5'>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox3"
                value="100,000"
                checked={step === '100,000'}
                onChange={handleStepChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox3">100k</label>
            </div>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox4"
                value="500,000"
                checked={step === '500,000'}
                onChange={handleStepChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox4">500k</label>
            </div>
          </div>
        </div>
        
        <div className='row mb-3'>
          <div className='col-4 text-end'>Reward Function</div>
          <div className='col-5'>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox5"
                value="simple"
                checked={rewardFunction === 'simple'}
                onChange={handleRewardChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox5">simple</label>
            </div>
            <div className="form-check form-check-inline">
              <input
                className="form-check-input"
                type="radio"
                id="inlineCheckbox6"
                value="complex"
                checked={rewardFunction === 'complex'}
                onChange={handleRewardChange}
              />
              <label className="form-check-label" htmlFor="inlineCheckbox6">complex</label>
            </div>
          </div>
        </div>
      </div>
        
      <div className="my-3">
        <button
          onClick={handleTrain}
          className="btn btn-secondary mx-2"
        >
          Run
        </button>
      </div>
      <div className="d-flex justify-content-center">
        {(videoReady) ? 
          <video
            src={`${endpoint}/video?${ts}`}
            controls
            autoPlay
            width="600"
          /> :
          <div className="spinner-border" role="status"></div>
        }
      </div> 
    </div>
  );
}
