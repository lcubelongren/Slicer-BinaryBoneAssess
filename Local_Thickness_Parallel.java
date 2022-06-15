/*
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2006 - 2020 Fiji developers.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

package sc.fiji.localThickness;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;

/* Bob Dougherty September 6, 2006

Input: Distance Ridge (32-bit stack) (Output from Distance Ridge.java)
Output: Local Thickness.  Overwrites the input.
Reference: T. Holdegrand and P. Ruegsegger, "A new method for the model-independent assessment of
thickness in three-dimensional images," Journal of Microscopy, Vol. 185 Pt. 1, January 1997 pp 67-75.

Version 1: September 6, 2006.
Version 2: September 25, 2006.  Fixed several bugs that resulted in
                                non-symmetrical output from symmetrical input.
Version 2.1 Oct. 1, 2006.  Fixed a rounding error that caused some points to be missed.
Version 3 July 31, 2007.  Parellel processing version.
Version 3.1  Multiplies the output by 2 to conform with the definition of local thickness


 License:
	Copyright (c) 2006, OptiNav, Inc.
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

		Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
		Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
		Neither the name of OptiNav, Inc. nor the names of its contributors
	may be used to endorse or promote products derived from this software
	without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
	A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
public class Local_Thickness_Parallel implements PlugInFilter {

	private ImagePlus imp;
	private ImagePlus resultImage;
	public float[][] data;
	public int w, h, d;
	public boolean runSilent = false;

	@Override
	public int setup(final String arg, final ImagePlus imp) {
		this.imp = imp;
		return DOES_32;
	}

	@Override
	public void run(final ImageProcessor ip) {
		resultImage = imp.duplicate();
		final ImageStack stack = resultImage.getStack();

		w = stack.getWidth();
		h = stack.getHeight();
		d = resultImage.getStackSize();
		final int wh = w * h;
		// Create reference to input data
		final float[][] s = new float[d][];
		for (int k = 0; k < d; k++)
			s[k] = (float[]) stack.getPixels(k + 1);
		float[] sk;
		// Count the distance ridge points on each slice
		final int[] nRidge = new int[d];
		int ind, nr, iR;
		IJ.showStatus("Local Thickness: scanning stack ");
		for (int k = 0; k < d; k++) {
			sk = s[k];
			nr = 0;
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					ind = i + w * j;
					if (sk[ind] > 0) nr++;
				}
			}
			nRidge[k] = nr;
		}
		final int[][] iRidge = new int[d][];
		final int[][] jRidge = new int[d][];
		final float[][] rRidge = new float[d][];
		// Pull out the distance ridge points
		int[] iRidgeK, jRidgeK;
		float[] rRidgeK;
		float sMax = 0;
		for (int k = 0; k < d; k++) {
			nr = nRidge[k];
			iRidge[k] = new int[nr];
			jRidge[k] = new int[nr];
			rRidge[k] = new float[nr];
			sk = s[k];
			iRidgeK = iRidge[k];
			jRidgeK = jRidge[k];
			rRidgeK = rRidge[k];
			iR = 0;
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					ind = i + w * j;
					if (sk[ind] > 0) {
						;
						iRidgeK[iR] = i;
						jRidgeK[iR] = j;
						rRidgeK[iR++] = sk[ind];
						if (sk[ind] > sMax) sMax = sk[ind];
						sk[ind] = 0;
					}
				}
			}
		}
		final int nThreads = Runtime.getRuntime().availableProcessors();
		final Object[] resources = new Object[d];// For synchronization
		for (int k = 0; k < d; k++) {
			resources[k] = new Object();
		}
		final LTThread[] ltt = new LTThread[nThreads];
		for (int thread = 0; thread < nThreads; thread++) {
			ltt[thread] = new LTThread(thread, nThreads, w, h, d, nRidge, s, iRidge,
				jRidge, rRidge, resources);
			ltt[thread].start();
		}
		try {
			for (int thread = 0; thread < nThreads; thread++) {
				ltt[thread].join();
			}
		}
		catch (final InterruptedException ie) {
			IJ.error("A thread was interrupted .");
		}

		// Fix the square values and apply factor of 2
		IJ.showStatus("Local Thickness: square root ");
		for (int k = 0; k < d; k++) {
			sk = s[k];
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					ind = i + w * j;
					sk[ind] = (float) (2 * Math.sqrt(sk[ind]));
				}
			}
		}
		IJ.showStatus("Local Thickness complete");

		final String title = stripExtension(imp.getTitle());
		resultImage.setTitle(title + "_LT");
		resultImage.getProcessor().setMinAndMax(0, sMax);

		if (!runSilent) {
			resultImage.show();
			IJ.run("Fire");
		}
	}

	// Modified from ImageJ code by Wayne Rasband
	String stripExtension(String name) {
		if (name != null) {
			final int dotIndex = name.lastIndexOf(".");
			if (dotIndex >= 0) name = name.substring(0, dotIndex);
		}
		return name;
	}

	public ImagePlus getResultImage() {
		return resultImage;
	}

	class LTThread extends Thread {

		int thread, nThreads, w, h, d, nR;
		float[][] s;
		int[] nRidge;
		int[][] iRidge, jRidge;
		float[][] rRidge;
		Object[] resources;

		public LTThread(final int thread, final int nThreads, final int w,
			final int h, final int d, final int[] nRidge, final float[][] s,
			final int[][] iRidge, final int[][] jRidge, final float[][] rRidge,
			final Object[] resources)
		{
			this.thread = thread;
			this.nThreads = nThreads;
			this.w = w;
			this.h = h;
			this.d = d;
			this.s = s;
			this.nRidge = nRidge;
			this.iRidge = iRidge;
			this.jRidge = jRidge;
			this.rRidge = rRidge;
			this.resources = resources;
		}

		@Override
		public void run() {
			int i, j;
			final float[] sk;
			float[] sk1;
			// Loop through ridge points. For each one, update the local thickness for
			// the points within its sphere.
			float r;
			int rInt, ind1;
			int iStart, iStop, jStart, jStop, kStart, kStop;
			float r1SquaredK, r1SquaredJK, r1Squared, s1;
			int rSquared;
			int[] iRidgeK, jRidgeK;
			float[] rRidgeK;
			for (int k = thread; k < d; k += nThreads) {
				IJ.showStatus("Local Thickness: processing slice " + (k + 1) + "/" +
					(d + 1));
				final int nR = nRidge[k];
				iRidgeK = iRidge[k];
				jRidgeK = jRidge[k];
				rRidgeK = rRidge[k];
				// sk = s[k];
				for (int iR = 0; iR < nR; iR++) {
					i = iRidgeK[iR];
					j = jRidgeK[iR];
					r = rRidgeK[iR];
					rSquared = (int) (r * r + 0.5f);
					rInt = (int) r;
					if (rInt < r) rInt++;
					iStart = i - rInt;
					if (iStart < 0) iStart = 0;
					iStop = i + rInt;
					if (iStop >= w) iStop = w - 1;
					jStart = j - rInt;
					if (jStart < 0) jStart = 0;
					jStop = j + rInt;
					if (jStop >= h) jStop = h - 1;
					kStart = k - rInt;
					if (kStart < 0) kStart = 0;
					kStop = k + rInt;
					if (kStop >= d) kStop = d - 1;
					for (int k1 = kStart; k1 <= kStop; k1++) {
						r1SquaredK = (k1 - k) * (k1 - k);
						sk1 = s[k1];
						for (int j1 = jStart; j1 <= jStop; j1++) {
							r1SquaredJK = r1SquaredK + (j1 - j) * (j1 - j);
							if (r1SquaredJK <= rSquared) {
								for (int i1 = iStart; i1 <= iStop; i1++) {
									r1Squared = r1SquaredJK + (i1 - i) * (i1 - i);
									if (r1Squared <= rSquared) {
										ind1 = i1 + w * j1;
										s1 = sk1[ind1];
										if (rSquared > s1) {
											// Get a lock on sk1 and check again to make sure
											// that another thread has not increased
											// sk1[ind1] to something larger than rSquared.
											// A test shows that this may not be required...
											synchronized (resources[k1]) {
												s1 = sk1[ind1];
												if (rSquared > s1) {
													sk1[ind1] = rSquared;
												}
											}
										}
									} // if within shere of DR point
								} // i1
							} // if k and j components within sphere of DR point
						} // j1
					} // k1
				} // iR
			} // k
		}// run
	}// Step1Thread
}
