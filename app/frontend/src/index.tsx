import appLogo from "./assets/munro-logo.png";
import React from "react";
import ReactDOM from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";
import { I18nextProvider } from "react-i18next";
import { HelmetProvider } from "react-helmet-async";
import { initializeIcons } from "@fluentui/react";

import "./index.css";

import Chat from "./pages/chat/Chat";
import LayoutWrapper from "./layoutWrapper";
import i18next from "./i18n/config";

initializeIcons();

const uiLogo = appLogo;
const uiTitle = import.meta.env.VITE_UI_TITLE;

const router = createHashRouter([
    {
        path: "/",
        element: <LayoutWrapper />, // ✅ No props needed here
        children: [
            {
                index: true,
                element: <Chat />
            },
            {
                path: "qa",
                lazy: async () => ({
                  Component: (await import("./pages/ask/Ask")).default
                })
            },
            {
                path: "*",
                lazy: async () => ({
                  Component: (await import("./pages/NoPage")).default
                })
            }
        ]
    }
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
        <I18nextProvider i18n={i18next}>
            <HelmetProvider>
                <RouterProvider router={router} />
            </HelmetProvider>
        </I18nextProvider>
    </React.StrictMode>
);
