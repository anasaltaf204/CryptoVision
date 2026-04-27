/**
 * components/ProtectedRoute.jsx
 *
 * Changes from original:
 *  - Fixed a bug: the original returned <LoadingSpinner /> (undefined component)
 *    when currentUser was null during the Firebase auth initialisation phase.
 *    Now it returns null during loading (AuthContext already blocks render via
 *    `!loading && children`, so this branch is only reached post-init).
 *  - Updated import path to the new context/ location.
 */

import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ children }) {
    const { currentUser, loading } = useAuth();

    // AuthContextProvider already suppresses children until Firebase resolves,
    // but guard here too in case this component is ever used standalone.
    if (loading) return null;

    if (!currentUser) {
        return <Navigate to="/auth" replace />;
    }

    return children;
}
