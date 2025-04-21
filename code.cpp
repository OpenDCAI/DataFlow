/*
 *
 * ©K. D. Hedger. Fri 21 Aug 12:29:04 BST 2015 keithdhedger@gmail.com

 * This file (LFSTK_lib.h) is part of LFSToolKit.

 * LFSToolKit is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * at your option) any later version.

 * LFSToolKit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with LFSToolKit.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Portions of this file is from here:
 * https://github.com/rahra/cairo_jpg
 * Many thanks!
 */

#ifndef _LFSTK_LIB_
#define _LFSTK_LIB_

#include <Imlib2.h>

#define	MAXVARS 100
#define MAXBUFFER 512

enum	{TYPEWINDOW=0,TYPEWINDOWTILE,TYPEBUTTON,TYPEBUTTONTILE,TYPEMENUITEM,TYPEMENUITEMTILE,TYPEFONTCOLOUR,TYPEMENUITEMFONTCOLOUR,TYPEMENUITEMFONT,TYPEFONT,TYPEMONOFONT,TYPECURSORCOLOUR,TYPESBTROUGHCOLOUR,TYPELISTTROUGHCOLOUR};

/**
 * \brief Library of useful routines for LFSToolKit.
*/
class LFSTK_lib
{
	public:
		LFSTK_lib(bool loadvars);
		~LFSTK_lib();

		const char		*LFSTK_getGlobalString(int state,int type);
		void				LFSTK_setGlobalString(int state,int type,const char *str);
		bool				LFSTK_getAutoLabelColour(void);
		void				LFSTK_setAutoLabelColour(bool toset);
		bool				LFSTK_getUseTheme(void);
		void				LFSTK_setUseTheme(bool use);
		const char		*LFSTK_getThemePath(void);
		void				LFSTK_setThemePath(char *path);

		const char		*bestFontColour(long pixel);
		static bool		LFSTK_gadgetEvent(void *self,XEvent *e,int type);

		char				*LFSTK_findThemedIcon(const char *theme,const char *icon,const char *catagory);
		bool				LFSTK_pointInRect(pointStruct *point,geometryStruct *geom);
		unsigned long	LFSTK_getColourFromName(Display *display,Colormap cm,const char *name);

//graphics
		void				LFSTK_setCairoSurface(Display *display,Window window,Visual *visual,cairo_surface_t **sfc,cairo_t **cr,int width,int height);
		cairo_surface_t	*LFSTK_createSurfaceFromPath(const char *path);
		cairo_surface_t	*LFSTK_cairo_image_surface_create_from_jpeg(const char *filename);
		Pixmap			LFSTK_getWindowPixmap(Display *display,Window win);

//files
		char				*LFSTK_oneLiner(const char* fmt,...);
		std::string		LFSTK_oneLiner(const std::string fmt,...);
		char				*LFSTK_getMimeType(const char* path);
		std::string		LFSTK_grepInFile(const std::string filepath,const std::string needle);
		std::string		LFSTK_getNthNeedle(const std::string haystack,int needlecnt,const std::string delimiter=" ");
		void				LFSTK_getFileInfo(const char* path,fileInformation* info);

//colours and prefs
		void				LFSTK_setColourFromName(Display *display,Colormap cm,colourStruct *colptr,const char *name);
		void				LFSTK_reloadPrefs(void);
		void				LFSTK_loadDesktopIconTheme(void);

//utils
		char				*LFSTK_cleanString(const char *str);
//messaging		
//debug
//
//new prefs
		LFSTK_prefsClass	prefs;
		std::string		desktopIconTheme="gnome";
	private:
		cairo_surface_t	*cairo_image_surface_create_from_jpeg_mem(const unsigned char* data, size_t len);
};
#endif



