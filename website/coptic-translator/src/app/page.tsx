"use client";
import TranslationComponent from "./components/translation"; // Adjust the path accordingly
import Info from "./components/info"; // Adjust the path accordingly
import Header from "./components/header";
const Page = () => {
  return (
    <div className="flex flex-col items-center bg-egyptian min-h-screen p-8 text-teal">
      <Header />
      <TranslationComponent />
      <Info title="Authors">
        Andrew Megalaa
        <br />
        Maxim Enis
      </Info>
      <Info title="Acknowledgements">
        Caroline T. Schroeder
        <br />
        Amir Zeldes
      </Info>
      <Info title="Data">
        Caroline T. Schroeder, Amir Zeldes, et al., Coptic SCRIPTORIUM,
        2013-2023, http://copticscriptorium.org.
      </Info>
      <Info title="Sources and Licenses">
        All the documents used for training are licensed{" "}
        <a
          href="https://creativecommons.org/licenses/by/3.0/us/"
          target="_blank"
          rel="noopener noreferrer"
        >
          CC-BY 3.0{" "}
        </a>
        or
        <a
          href="https://creativecommons.org/licenses/by/4.0/"
          target="_blank"
          rel="noopener noreferrer"
        >
          {" "}
          4.0
        </a>{" "}
        unless otherwise indicated.
        <br />
        <br />
        Major exceptions include:
        <br />
        <div className="ml-4">
          <a
            href="http://www.copticscriptorium.org/download/corpora/Mark/coptic_nt_sahidic.html"
            target="_blank"
            rel="noopener noreferrer"
          >
            Sahidica New Testament specific license
          </a>
          <br />
          <a
            href="https://creativecommons.org/licenses/by-sa/3.0/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Canons of Apa Johannes CC-BY-SA 3.0
          </a>
          <br />
          <a
            href="https://creativecommons.org/licenses/by-sa/4.0/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Sahidic Old Testament CC-BY-SA 4.0
          </a>
        </div>
        <br />
        Individual files on the Scriptorium also contain licensing information.
      </Info>
    </div>
  );
};

export default Page;
