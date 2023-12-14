import React, { useState, ChangeEvent, use, useEffect } from "react";
import { FaExchangeAlt } from "react-icons/fa";
// import css
import "./translation.css";

const ENGLISH_API = "/api/predictions/english";
const COPTIC_API = "/api/predictions/coptic";
const DELAY = 500;
const regexEnglish = /^[a-zA-Z\s.,!?'"-]*$/;
const regexCoptic = /^[\u2C80-\u2CFF\u03E2-\u03EF\s.,!?'"-]*$/;

const TranslationComponent: React.FC = () => {
  const [srcText, setSrcText] = useState<string>("");
  const [tgtText, setTgtText] = useState<string>("");
  const [tgtTextLoading, setTgtTextLoading] = useState<boolean>(false);
  const [isEnglishToCoptic, setIsEnglishToCoptic] = useState<boolean>(true);
  const srcTextRef = React.useRef<HTMLTextAreaElement>(null);
  const tgtTextRef = React.useRef<HTMLTextAreaElement>(null);

  const placeholderText = (isEnglishToCoptic: boolean) => {
    return `Type or paste ${
      isEnglishToCoptic ? "English" : "Coptic"
    } text here...`;
  };

  useEffect(() => {
    let translationTimeout: NodeJS.Timeout;
    const controller = new AbortController();
    const signal = controller.signal;
    translationTimeout = setTimeout(() => {
      const api = isEnglishToCoptic ? COPTIC_API : ENGLISH_API;
      if (srcText === "") {
        setTgtText("");
        return;
      }
      setTgtTextLoading(true);

      fetch(api, {
        method: "POST",
        headers: {
          "Content-Type": "text/plain",
        },
        body: srcText,
        signal: signal,
      })
        .then((response) => response.json())
        .then((response) => {
          if (!response.translation) {
            throw new Error("Translation failed");
          }
          setTgtText(response.translation);
        })
        .catch((error) => console.error(error))
        .finally(() => setTgtTextLoading(false));
    }, DELAY);
    return () => {
      clearTimeout(translationTimeout);
      controller.abort();
      setTgtTextLoading(false);
    };
  }, [srcText, isEnglishToCoptic]);

  useEffect(() => {
    const tempText = srcText;
    setSrcText(tgtText);
    setTgtText(tempText);
  }, [isEnglishToCoptic]);

  const handleSrcTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    if (isEnglishToCoptic) {
      if (regexEnglish.test(newText)) {
        setSrcText(newText);
      }
    } else {
      if (regexCoptic.test(newText)) {
        setSrcText(newText);
      }
    }
  };

  const autoResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (srcTextRef.current && tgtTextRef.current) {
      srcTextRef.current.style.height = `${e.target.scrollHeight}px`;
      tgtTextRef.current.style.height = `${e.target.scrollHeight}px`;
    }
  };

  return (
    <div className="flex flex-col items-center bg-egyptian min-h-screen p-8 text-teal">
      <div className="w-full flex justify-center text-scriptorium-red">
        <div className="w-1/2 pr-4">
          <h2 className="text-2xl mb-4 text-center font-hieroglyph">
            {isEnglishToCoptic ? "English" : "Coptic"}
          </h2>
          <textarea
            ref={srcTextRef}
            className="border p-2 w-full bg-scriptorium-red-left text-teal rounded-lg"
            onChange={(e) => {
              handleSrcTextChange(e);
              autoResize(e);
            }}
            value={srcText}
            placeholder={placeholderText(isEnglishToCoptic)}
            style={{
              minHeight: "10rem",
              resize: "none",
              overflow: "hidden",
              outline: "none",
            }}
          />
        </div>
        <div className="flex flex-col items-center justify-center">
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
            ref={tgtTextRef}
            className="border p-2 w-full bg-scriptorium-red-right text-teal rounded-lg no-highlights"
            value={tgtTextLoading ? tgtText + "..." : tgtText}
            readOnly={true}
            style={{
              minHeight: "10rem",
              resize: "none",
              overflow: "hidden",
              outline: "none",
              border: "none",
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default TranslationComponent;
