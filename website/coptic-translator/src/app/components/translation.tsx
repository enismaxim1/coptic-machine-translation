import React, { useState, ChangeEvent, use, useEffect } from "react";
import { FaExchangeAlt } from "react-icons/fa";
// import css
import "./translation.css";

const ENGLISH_API = "/api/predictions/my_tc";
const COPTIC_API = "/api/predictions/my_tc";

const TranslationComponent: React.FC = () => {
  const [srcText, setSrcText] = useState<string>("");
  const [tgtText, setTgtText] = useState<string>("");
  const [tgtTextLoading, setTgtTextLoading] = useState<boolean>(false);
  const [isEnglishToCoptic, setIsEnglishToCoptic] = useState<boolean>(true);

  const placeholderText = (isEnglishToCoptic: boolean) => {
    return `Type or paste ${
      isEnglishToCoptic ? "English" : "Coptic"
    } text here...`;
  };

  useEffect(() => {
    let api = ENGLISH_API;
    if (isEnglishToCoptic) {
      api = COPTIC_API;
    }
    if (srcText === "") {
      return;
    }
    setTgtTextLoading(true);
    fetch(api, {
      method: "POST",
      headers: {
        "Content-Type": "text/plain",
      },
      body: srcText,
    })
      .then((response) => response.text())
      .then((response) => {
        setTgtText(response);
      })
      .catch((error) => console.error(error))
      .finally(() => setTgtTextLoading(false));
  }, [srcText, isEnglishToCoptic]);

  useEffect(() => {
    const tempText = srcText;
    setSrcText(tgtText);
    setTgtText(tempText);
  }, [isEnglishToCoptic]);

  return (
    <div className="flex flex-col items-center bg-egyptian min-h-screen p-8 text-teal">
      <div className="w-full flex justify-center text-scriptorium-red">
        <div className="w-1/2 pr-4">
          <h2 className="text-2xl mb-4 text-center font-hieroglyph">
            {isEnglishToCoptic ? "English" : "Coptic"}
          </h2>
          <textarea
            className="border p-2 w-full h-40 bg-scriptorium-red-left text-teal rounded-lg"
            onChange={(e) => setSrcText(e.target.value)}
            value={srcText}
            placeholder={placeholderText(isEnglishToCoptic)}
            // set outline color
            style={{ outline: "none" }}
          />
        </div>
        <div className="flex flex-col items-center justify-center">
          <button
            // onClick={handleTranslateClick}
            className="p-2 bg-scriptorium-red-left text-teal rounded-pyramid shadow-md mb-4 font-hieroglyph"
          >
            Translate
          </button>
          <button
            onClick={() => setIsEnglishToCoptic(!isEnglishToCoptic)}
            className="p-2 bg-scriptorium-red-left text-teal rounded-pyramid shadow-md"
          >
            <FaExchangeAlt size={24} />
          </button>
        </div>
        <div className="w-1/2 pl-4">
          <h2 className="text-2xl mb-4 text-center text-scriptorium-red">
            {isEnglishToCoptic ? "Coptic" : "English"}
          </h2>
          <textarea
            className="border p-2 w-full h-40 bg-scriptorium-red-right text-teal rounded-lg no-highlights"
            value={tgtTextLoading ? tgtText + "..." : tgtText}
            readOnly={true}
            style={{ outline: "none" }}
          />
        </div>
      </div>
    </div>
  );
};

export default TranslationComponent;
