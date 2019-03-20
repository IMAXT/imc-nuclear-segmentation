class Grid:
    """[summary]

        Parameters
        ----------
        img : [type]
            [description]
        tile_h : int, optional
            [description] (the default is 1000, which [default_description])
        tile_w : int, optional
            [description] (the default is 1000, which [default_description])
        tileOverlap_h : int, optional
            [description] (the default is 25, which [default_description])
        tileOverlap_w : int, optional
            [description] (the default is 25, which [default_description])
    """

    # By Ali:

    # A class to generate positional information in order
    # to split a large image into smaller tiles/chunks.

    # Desing
    # (A) = chunksOverlap_pt1 :(x_overlapTopLeft                  , y_overlapTopLeft                   )
    # (B) = chunks_pt1        :(x_topLeft                         , y_topLeft                          )
    # (C) = chunks_pt2        :(x_topLeft        + ch_width       , y_topLeft        + ch_height       )
    # (D) = chunksOverlap_pt2 :(x_overlapTopLeft + ch_overlapWidth, y_overlapTopLeft + ch_overlapHeight)

    #               <================= ch_overlapWidth =================>

    #          d_overlapLeft_w            ch_width             d_overlapRight_w
    #               <====> <=====================================> <====>

    #              (A)
    #               .....................................................
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .......---------------------------------------.......
    #               .     |(B)                                    |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .     |                                   (C) |     .
    #               ......---------------------------------------........
    #               .     |                                       |     .
    #               .     |                                       |     .
    #               .....................................................
    #                                                                   (D)

    def __init__(
        self, img, tile_h=1000, tile_w=1000, tileOverlap_h=25, tileOverlap_w=25
    ):
        # Default tile dimension is 1000 x 1000, i.e. 1.0 Megapixel
        # with 25 pixels overlap along each side

        # input matter
        self.img = img
        self.chunk_w = tile_w
        self.chunk_h = tile_h
        self.chunkOverlap_w = tileOverlap_w
        self.chunkOverlap_h = tileOverlap_h
        self.img_h = img.shape[0]
        self.img_w = img.shape[1]
        self.n_chunk_h = int(self.img_h / self.chunk_h) + 1
        self.n_chunk_w = int(self.img_w / self.chunk_w) + 1
        self.n_chunk_total_complete = (self.n_chunk_h - 1) * (self.n_chunk_w - 1)
        self.n_chunk_total = self.n_chunk_h * self.n_chunk_w

    def _grid_parameters(self):

        chunks_id = []
        chunks_topLeft_x = []
        chunks_topLeft_y = []
        chunks_height = []
        chunks_width = []
        chunks_pt1 = []
        chunks_pt2 = []

        chunksOverlap_topLeft_x = []
        chunksOverlap_topLeft_y = []
        chunksOverlap_height = []
        chunksOverlap_width = []
        chunksOverlap_pt1 = []
        chunksOverlap_pt2 = []

        n_chunk = 0  # initialise a chunk ID

        # iterate (up -> down)
        for j in range(self.n_chunk_h):

            # iterate (left -> right)
            for i in range(self.n_chunk_w):

                n_chunk += 1

                #  --------------
                # | Chunk itself |
                #  --------------
                # ===================================================================

                # adjust for the last chunk width along 'w' direction (left -> right)
                # -------------------------------------------------------------------

                # check if we reach to the last chunk on the right side of the image
                if (i * self.chunk_w + self.chunk_w) > self.img_w:
                    final_chunk_w = self.img_w - ((i + 1) * self.chunk_w) - 1
                else:
                    final_chunk_w = 0

                # adjust for the last chunk height along 'h' direction (up -> down)
                # -----------------------------------------------------------------

                # check if we reach to the last chunk on the bottom of the image
                if (j * self.chunk_h + self.chunk_h) > self.img_h:
                    final_chunk_h = self.img_h - ((j + 1) * self.chunk_h) - 1
                else:
                    final_chunk_h = 0

                # output info
                chunks_id.append(n_chunk)

                x_topLeft, y_topLeft = i * self.chunk_w, j * self.chunk_h
                ch_width, ch_height = (
                    (self.chunk_w + final_chunk_w),
                    (self.chunk_h + final_chunk_h),
                )

                chunks_topLeft_x.append(x_topLeft)
                chunks_topLeft_y.append(y_topLeft)

                chunks_width.append(ch_width)
                chunks_height.append(ch_height)

                chunks_pt1.append((x_topLeft, y_topLeft))
                chunks_pt2.append((x_topLeft + ch_width, y_topLeft + ch_height))

                #  ---------------
                # | Chunk overlap |
                #  ---------------
                # ===================================================================

                # stage.1: Determining d_overlap Left, Right, Top, and Bottom

                # extereme left overlap
                if (x_topLeft - self.chunkOverlap_w) <= 0:
                    d_overlapLeft_w = 0
                else:
                    d_overlapLeft_w = self.chunkOverlap_w

                # extereme right overlap
                if (x_topLeft + ch_width + self.chunkOverlap_w) >= self.img_w:
                    d_overlapRight_w = 0
                else:
                    d_overlapRight_w = self.chunkOverlap_w

                # extereme top
                if (y_topLeft - self.chunkOverlap_h) <= 0:
                    d_overlapTop_h = 0
                else:
                    d_overlapTop_h = self.chunkOverlap_h

                # extereme bottom
                if (y_topLeft + ch_height + self.chunkOverlap_h) >= self.img_h:
                    d_overlapBottom_h = 0
                else:
                    d_overlapBottom_h = self.chunkOverlap_h

                # stage.2: chunksOverlap information

                x_overlapTopLeft = x_topLeft - d_overlapLeft_w
                y_overlapTopLeft = y_topLeft - d_overlapTop_h
                ch_overlapWidth = d_overlapLeft_w + ch_width + d_overlapRight_w
                ch_overlapHeight = d_overlapTop_h + ch_height + d_overlapBottom_h

                chunksOverlap_topLeft_x.append(x_overlapTopLeft)
                chunksOverlap_topLeft_y.append(y_overlapTopLeft)
                chunksOverlap_height.append(ch_overlapWidth)
                chunksOverlap_width.append(ch_overlapHeight)
                chunksOverlap_pt1.append((x_overlapTopLeft, y_overlapTopLeft))
                chunksOverlap_pt2.append(
                    (
                        x_overlapTopLeft + ch_overlapWidth,
                        y_overlapTopLeft + ch_overlapHeight,
                    )
                )

        return (
            chunks_id,
            chunks_topLeft_x,
            chunks_topLeft_y,
            chunks_height,
            chunks_width,
            chunks_pt1,
            chunks_pt2,
            chunksOverlap_topLeft_x,
            chunksOverlap_topLeft_y,
            chunksOverlap_height,
            chunksOverlap_width,
            chunksOverlap_pt1,
            chunksOverlap_pt2,
        )

    def get_chunks_id(self):
        return self._grid_parameters()[0]

    def get_chunks_topLeft_x(self):
        return self._grid_parameters()[1]

    def get_chunks_topLeft_y(self):
        return self._grid_parameters()[2]

    def get_chunks_height(self):
        return self._grid_parameters()[3]

    def get_chunks_width(self):
        return self._grid_parameters()[4]

    def get_chunks_pt1(self):
        return self._grid_parameters()[5]

    def get_chunks_pt2(self):
        return self._grid_parameters()[6]

    def get_chunksOverlap_topLeft_x(self):
        return self._grid_parameters()[7]

    def get_chunksOverlap_topLeft_y(self):
        return self._grid_parameters()[8]

    def get_chunksOverlap_height(self):
        return self._grid_parameters()[9]

    def get_chunksOverlap_width(self):
        return self._grid_parameters()[10]

    def get_chunksOverlap_pt1(self):
        return self._grid_parameters()[11]

    def get_chunksOverlap_pt2(self):
        return self._grid_parameters()[12]
