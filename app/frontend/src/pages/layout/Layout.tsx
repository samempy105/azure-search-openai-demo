import React, { useState, useEffect, useRef, RefObject } from "react";
import { Outlet, NavLink, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import styles from "./Layout.module.css";

import { useLogin } from "../../authConfig";

import { LoginButton } from "../../components/LoginButton";
import { IconButton } from "@fluentui/react";

interface LayoutProps {
    uiTitle?: string;
    uiLogo?: string;
}

const Layout = ({ uiTitle, uiLogo }: LayoutProps) => {
    const { t } = useTranslation();
    const [menuOpen, setMenuOpen] = useState(false);
    const menuRef: RefObject<HTMLDivElement> = useRef(null);

    const toggleMenu = () => {
        setMenuOpen(!menuOpen);
    };

    const handleClickOutside = (event: MouseEvent) => {
        if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
            setMenuOpen(false);
        }
    };

    useEffect(() => {
        if (menuOpen) {
            document.addEventListener("mousedown", handleClickOutside);
        } else {
            document.removeEventListener("mousedown", handleClickOutside);
        }
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [menuOpen]);

    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer} ref={menuRef}>
                    <Link to="/" className={styles.headerTitleContainer}>
                        {uiLogo && <img src={uiLogo} alt="logo" height="32" />}
                        <h3 className={styles.headerTitle}>{uiTitle || t("headerTitle")}</h3>
                    </Link>

                    
                    <div className={styles.loginMenuContainer}>
                        {useLogin && <LoginButton />}
                        <IconButton
                            iconProps={{ iconName: "GlobalNavButton" }}
                            className={styles.menuToggle}
                            onClick={toggleMenu}
                            ariaLabel={t("labels.toggleMenu")}
                        />
                    </div>
                </div>
            </header>

            <Outlet />
        </div>
    );
};

export default Layout;
