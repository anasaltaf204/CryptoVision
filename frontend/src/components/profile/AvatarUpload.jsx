/**
 * components/profile/AvatarUpload.jsx
 *
 * Reusable avatar upload component.
 * - Click to open file picker (accepts image/*)
 * - Validates file type and size (max 500 KB)
 * - Converts the file to a base64 data-URL via FileReader
 * - Calls onUpload(dataUrl) when ready — parent owns the save logic
 * - Shows a preview overlay on hover with a camera icon
 */

import React, { useRef, useState } from "react";

const MAX_SIZE_BYTES = 500 * 1024; // 500 KB

export default function AvatarUpload({ currentURL, displayName, onUpload, size = 96 }) {
    const inputRef           = useRef(null);
    const [preview, setPreview] = useState(null);
    const [error, setError]   = useState(null);

    function handleFileChange(e) {
        const file = e.target.files[0];
        if (!file) return;
        setError(null);

        if (!file.type.startsWith("image/")) {
            setError("Please select an image file.");
            return;
        }
        if (file.size > MAX_SIZE_BYTES) {
            setError("Image must be smaller than 500 KB.");
            return;
        }

        const reader = new FileReader();
        reader.onload = (ev) => {
            const dataUrl = ev.target.result;
            setPreview(dataUrl);
            onUpload(dataUrl);
        };
        reader.readAsDataURL(file);
    }

    const displaySrc = preview || currentURL;
    const initials   = displayName?.[0]?.toUpperCase() || "U";

    return (
        <div className="avatar-upload" style={{ "--avatar-size": `${size}px` }}>
            <button
                type="button"
                className="avatar-upload__trigger"
                onClick={() => inputRef.current?.click()}
                aria-label="Upload profile photo"
            >
                {displaySrc ? (
                    <img src={displaySrc} alt={displayName} className="avatar-upload__img" />
                ) : (
                    <div className="avatar-upload__initials">{initials}</div>
                )}
                <div className="avatar-upload__overlay">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                    <span>Change Photo</span>
                </div>
            </button>
            <input
                ref={inputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={handleFileChange}
            />
            {error && <p className="avatar-upload__error">{error}</p>}
        </div>
    );
}
