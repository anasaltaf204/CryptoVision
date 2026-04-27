/**
 * store/index.js
 *
 * Added userReducer to the store.
 */

import { configureStore } from "@reduxjs/toolkit";
import dashboardReducer from "./slices/dashboardSlice";
import userReducer      from "./slices/userSlice";

const store = configureStore({
    reducer: {
        dashboard: dashboardReducer,
        user:      userReducer,
    },
});

export default store;
