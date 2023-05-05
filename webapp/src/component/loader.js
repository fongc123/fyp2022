import React from 'react';
import Lottie from "lottie-react";
import loading from './99109-loading.json'
const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: loading.default,
    rendererSetting:{
        preserveAspectRatio: 'xMidYMid slice'
    }
};

export const Loading = () => {
    return(
        <div style={{marginTop: '10rem'}}>
            <Lottie animationData={loading} />
        </div>
    )
}
export default Loading;