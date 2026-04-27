/**
 * main.jsx
 *
 * Changes from original:
 *  - Wraps the app in Redux <Provider> so all components can access the store.
 *  - Provider is placed outside App so the store is available to any component
 *    in the tree, including those that also use AuthContext.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import store from "./store/index";
import App from "./App";
import "bootstrap/dist/css/bootstrap.min.css";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <Provider store={store}>
            <App />
        </Provider>
    </React.StrictMode>
);
