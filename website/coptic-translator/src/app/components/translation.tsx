import React, { useState, ChangeEvent } from "react";
import { FaExchangeAlt } from "react-icons/fa";
// import css
import "./translation.css";

const TranslationComponent: React.FC = () => {
  const [englishText, setEnglishText] = useState<string>("");
  const [copticText, setCopticText] = useState<string>("");
  const [isEnglishToCoptic, setIsEnglishToCoptic] = useState<boolean>(true);
  const [leftBoxHeader, setLeftBoxHeader] = useState<string>("English");
  const [rightBoxHeader, setRightBoxHeader] = useState<string>("Coptic");

  const placeholderText = (language: string) => {
    return `Type or paste ${language} text here...`;
  };

  const getEnglishTranslation = (word: string) => {
    return "english word";
  };

  const getCopticTranslation = (word: string) => {
    return "coptic word";
  };

  const translateText = (isEnglishToCoptic: boolean) => {
    if (isEnglishToCoptic) {
      setCopticText(getCopticTranslation(englishText));
    } else {
      setEnglishText(getEnglishTranslation(copticText));
    }
  };

  const handleTranslateClick = () => {
    translateText(isEnglishToCoptic);
  };

  const handleReverseTranslation = () => {
    setIsEnglishToCoptic(!isEnglishToCoptic);
    setBoxHeaders(!isEnglishToCoptic);
    // setEnglishText(isEnglishToCoptic ? copticText : "");
    // setCopticText(isEnglishToCoptic ? "" : englishText);
    translateText(!isEnglishToCoptic);
  };

  const setBoxHeaders = (isEnglishToCoptic: boolean) => {
    setLeftBoxHeader(isEnglishToCoptic ? "English" : "Coptic");
    setRightBoxHeader(isEnglishToCoptic ? "Coptic" : "English");
  };

  const handleEnglishChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const inputText: string = e.target.value;
    setEnglishText(inputText);
  };

  const handleCopticChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const inputText: string = e.target.value;
    setCopticText(inputText);
  };

  return (
    <div className="flex flex-col items-center bg-egyptian min-h-screen p-8 text-teal">
      <div className="w-full flex justify-center text-scriptorium-red">
        <div className="w-1/2 pr-4">
          <h2 className="text-2xl mb-4 text-center font-hieroglyph">
            {leftBoxHeader}
          </h2>
          <textarea
            className="border p-2 w-full h-40 bg-scriptorium-red-left text-teal rounded-lg"
            value={isEnglishToCoptic ? englishText : copticText}
            onChange={isEnglishToCoptic ? handleEnglishChange : handleCopticChange}
            placeholder={placeholderText(leftBoxHeader)}
            // set outline color 
            style={{outline: "none"}}
          />
        </div>
        <div className="flex flex-col items-center justify-center">
          <button
            onClick={handleTranslateClick}
            className="p-2 bg-scriptorium-red-left text-teal rounded-pyramid shadow-md mb-4 font-hieroglyph"
          >
            Translate
          </button>
          <button
            onClick={handleReverseTranslation}
            className="p-2 bg-scriptorium-red-left text-teal rounded-pyramid shadow-md"
          >
            <FaExchangeAlt size={24} />
          </button>
        </div>
        <div className="w-1/2 pl-4">
          <h2 className="text-2xl mb-4 text-center text-scriptorium-red">
            {rightBoxHeader}
          </h2>
          <textarea
            className="border p-2 w-full h-40 bg-scriptorium-red-right text-teal rounded-lg no-highlights"
            value={isEnglishToCoptic ? copticText : englishText}
            readOnly={true}  
            style={{outline: "none"}} 
          />
        </div>
      </div>
    </div>
  );
  
};

export default TranslationComponent;
